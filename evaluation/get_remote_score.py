import argparse
import json
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval import cal_em, cal_f1
from eval_g import cal_gen
from eval_r import cal_rsim
from tqdm import tqdm


def evaluate_one(d):
    try:
        generation = d['generation']
        answer_match = re.findall(r"<answer>(.*?)</answer>", generation, flags=re.DOTALL | re.IGNORECASE)
        answer = answer_match[-1].strip().replace("\n", " ") if answer_match else ""
        em_score = cal_em([d['golden_answers']], [answer])
        f1_score = cal_f1([d['golden_answers']], [answer])

        # 去重 context
        context = []
        for c in d['context']:
            if c not in context:
                context.append(c)

        rsim_score = cal_rsim(['\n'.join(context)], [d['knowledge']]) if d['knowledge'] != "" else 0.0
        gen_score = cal_gen(d['question'], d['golden_answers'], generation) if not d.get("skip_gen") else {"score": 0.0, "explanation": {}}

        d['em'] = em_score
        d['f1'] = f1_score
        d['rsim'] = rsim_score
        d['gen'] = gen_score["score"]
        d['gen_exp'] = gen_score["explanation"]

        return d
    except Exception as e:
        print(f"[ERROR] Failed processing sample: {d.get('question', 'N/A')}")
        traceback.print_exc()
        raise

def _step_from_name(path):
    match = re.search(r"step(\d+)", str(path))
    return int(match.group(1)) if match else -1


def _find_best_step(expr_dir):
    candidates = []
    for eval_file in sorted(Path(expr_dir).glob("evals_step*.json"), key=_step_from_name):
        data = json.loads(eval_file.read_text())
        f1_keys = [key for key in data if "f1" in key.lower()]
        score_keys = [key for key in data if "test_score" in key.lower()]
        metric_keys = f1_keys or score_keys
        if metric_keys:
            candidates.append((float(data[metric_keys[0]]), _step_from_name(eval_file)))
    if not candidates:
        raise RuntimeError(f"no eval metric found in {expr_dir}")
    return max(candidates)[1]


def _load_results(args):
    if args.results_file:
        results_file = Path(args.results_file)
    else:
        expr_dir = Path(args.dir)
        step = args.step if args.step >= 0 else _find_best_step(expr_dir)
        results_file = expr_dir / f"results_step{step}.json"
    xdata = json.loads(results_file.read_text())
    if args.limit > 0:
        xdata = xdata[: args.limit]
    data = []
    for x in xdata:
        knowledge = []
        ksplit = x['prediction'].split("</knowledge>")[:-1]
        for k in ksplit:
            knowledge.append('<knowledge>'.join(k.split("<knowledge>")[1:]))
        data.append({
            'question': x['question'],
            'golden_answers': x['golden_answers'],
            'context': x.get('context', []),
            'knowledge': '\n'.join(knowledge),
            'generation': x['prediction'],
            "format_score": x.get('format_score', 0),
            "turns": x.get('turns', 0),
            "skip_gen": args.skip_gen,
        })
    return results_file, data


def evaluate_method(args):
    dir = args.dir
    success_flag = False  # 控制是否成功保存
    method = args.results_file or dir

    try:
        results_file, data = _load_results(args)

        # 并行处理样本
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            data = list(tqdm(executor.map(evaluate_one, data), total=len(data), desc=method))

        # 汇总指标
        overall_em = sum([d['em'] for d in data]) / len(data)
        overall_f1 = sum([d['f1'] for d in data]) / len(data)
        overall_rsim = sum([d['rsim'] for d in data]) / len(data)
        overall_gen = sum([d['gen'] for d in data]) / len(data)

        print(f"{method} Overall EM: {overall_em:.4f}")
        print(f"{method} Overall F1: {overall_f1:.4f}")
        print(f"{method} Overall R-Sim: {overall_rsim:.4f}")
        print(f"{method} Overall Gen: {overall_gen:.4f}")

        save_base = args.out_dir or dir
        os.makedirs(save_base, exist_ok=True)

        result_path = os.path.join(save_base, "test_result.json")
        with open(result_path, 'w') as f:
            json.dump(data, f, indent=4)

        score_path = os.path.join(save_base, "test_score.json")
        with open(score_path, 'w') as f:
            json.dump({
                "results_file": str(results_file),
                "overall_em": overall_em,
                "overall_f1": overall_f1,
                "overall_rsim": overall_rsim,
                "overall_gen": overall_gen,
            }, f, indent=4)

        # 成功保存标志
        success_flag = True
        print(f"[SAVED] {result_path}")
        print(f"[SAVED] {score_path}")
        print(f"[SUCCESS] {method} finished and saved.")

    except Exception as e:
        print(f"\n[ERROR] {method} failed due to: {str(e)}")
        traceback.print_exc()
        raise

    if not success_flag:
        raise RuntimeError(f"{method} did not complete saving.")

    return True

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--dir', type=str, default='../expr_results/Qwen2.5-3B-Instruct_2WikiMultiHopQA_grpo')
    parse.add_argument('--results_file', type=str, default='')
    parse.add_argument('--out_dir', type=str, default='')
    parse.add_argument('--step', type=int, default=-1)
    parse.add_argument('--limit', type=int, default=-1)
    parse.add_argument('--workers', type=int, default=4)
    parse.add_argument('--skip_gen', action='store_true')

    args = parse.parse_args()
    evaluate_method(args)
