import os
import httpx
import time
import sys

MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/model/Qwen3-32B")
PLATFORM_URL = os.getenv("PLATFORM_URL", "http://10.0.0.1:8003")
TOKEN = os.getenv("TEAM_TOKEN", os.getenv("TOKEN", "4ab3e24be30209190cfd4a60d3fc22d7"))
TEAM_NAME = os.getenv("TEAM_NAME", "浙题会了")
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:9000")


def get_client():
    return httpx.Client(timeout=60.0)


client = get_client()


def compute_logprob(prompt, continuation):
    if not continuation:
        return 0.0

    full_text = prompt + continuation

    try:
        resp = client.post(f"{INFERENCE_URL}/v1/completions", json={
            "model": MODEL_PATH,
            "prompt": full_text,
            "max_tokens": len(continuation),
            "logprobs": 1,
            "temperature": 0.0,
        })

        if resp.status_code != 200:
            print(f"[ERROR] logprobs API error: {resp.status_code} {resp.text}")
            return 0.0

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return 0.0

        logprobs_data = choices[0].get("logprobs", {})
        token_logprobs = logprobs_data.get("token_logprobs", [])

        if not token_logprobs:
            return 0.0

        total_logprob = sum(token_logprobs)
        return total_logprob
    except Exception as e:
        print(f"[ERROR] compute_logprob exception: {e}")
        return 0.0


def compute_rolling_logprob(prompt):
    if not prompt:
        return 0.0

    try:
        resp = client.post(f"{INFERENCE_URL}/v1/completions", json={
            "model": MODEL_PATH,
            "prompt": prompt,
            "logprobs": 1,
            "temperature": 0.0,
        })

        if resp.status_code != 200:
            print(f"[ERROR] rolling logprobs API error: {resp.status_code} {resp.text}")
            return 0.0

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return 0.0

        logprobs_data = choices[0].get("logprobs", {})
        token_logprobs = logprobs_data.get("token_logprobs", [])

        if not token_logprobs:
            return 0.0

        total_logprob = sum(token_logprobs)
        return total_logprob
    except Exception as e:
        print(f"[ERROR] compute_rolling_logprob exception: {e}")
        return 0.0


def register():
    global client
    while True:
        try:
            print("[INFO] 正在注册到平台...")
            resp = client.post(f"{PLATFORM_URL}/register", json={"name": TEAM_NAME, "token": TOKEN})
            if resp.status_code == 200:
                result = resp.json()
                print(f"[INFO] 注册成功: {result}")
                return True
            else:
                print(f"[WARN] 注册失败，状态码: {resp.status_code}, 重试...")
        except httpx.ConnectError as e:
            print(f"[WARN] 注册连接失败: {e}, 重新创建连接...")
            client.close()
            client = get_client()
        except Exception as e:
            print(f"[WARN] 注册异常: {e}, 重试...")
        time.sleep(2)


def query_task():
    global client
    while True:
        try:
            resp = client.post(f"{PLATFORM_URL}/query", json={"token": TOKEN})
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"[WARN] query 失败，状态码: {resp.status_code}, 重试...")
        except httpx.ConnectError as e:
            print(f"[WARN] query 连接失败: {e}, 重新创建连接...")
            client.close()
            client = get_client()
        except Exception as e:
            print(f"[WARN] query 异常: {e}, 重试...")
        time.sleep(0.5)


def ask_task(task_id, sla):
    global client
    while True:
        try:
            resp = client.post(f"{PLATFORM_URL}/ask", json={
                "token": TOKEN, "task_id": task_id, "sla": sla
            })
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"[WARN] ask 失败，状态码: {resp.status_code}, 重试...")
        except httpx.ConnectError as e:
            print(f"[WARN] ask 连接失败: {e}, 重新创建连接...")
            client.close()
            client = get_client()
        except Exception as e:
            print(f"[WARN] ask 异常: {e}, 重试...")
        time.sleep(0.5)


def submit_task(task):
    global client
    while True:
        try:
            resp = client.post(f"{PLATFORM_URL}/submit", json={
                "user": {"name": TEAM_NAME, "token": TOKEN},
                "msg": task,
            })
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"[WARN] submit 失败，状态码: {resp.status_code}, 重试...")
        except httpx.ConnectError as e:
            print(f"[WARN] submit 连接失败: {e}, 重新创建连接...")
            client.close()
            client = get_client()
        except Exception as e:
            print(f"[WARN] submit 异常: {e}, 重试...")
        time.sleep(0.5)


def get_score():
    global client
    try:
        resp = client.get(f"{PLATFORM_URL}/leaderboard")
        if resp.status_code == 200:
            data = resp.json()
            leaderboard = data.get("leaderboard", [])
            for entry in leaderboard:
                if entry.get("name") == TEAM_NAME:
                    return entry
        return None
    except Exception as e:
        print(f"[WARN] 获取分数失败: {e}")
        return None


def do_inference(prompt, rt, gen_kwargs=None, continuation=None):
    global client

    if gen_kwargs is None:
        gen_kwargs = {}

    try:
        if rt == "generate_until":
            inf_resp = client.post(f"{INFERENCE_URL}/v1/completions", json={
                "model": MODEL_PATH,
                "prompt": prompt,
                "max_tokens": gen_kwargs.get("max_gen_toks", 256),
                "temperature": gen_kwargs.get("temperature", 0.0),
                "stop": gen_kwargs.get("until", []),
            })
            if inf_resp.status_code != 200:
                print(f"[ERROR] generate_until API error: {inf_resp.status_code} {inf_resp.text}")
                return {"response": ""}
            return {"response": inf_resp.json()["choices"][0]["text"]}

        elif rt == "loglikelihood":
            accuracy = compute_logprob(prompt, continuation)
            return {"accuracy": accuracy}

        elif rt == "loglikelihood_rolling":
            accuracy = compute_rolling_logprob(prompt)
            return {"accuracy": accuracy}

        else:
            print(f"[WARN] unknown eval_request_type: {rt}")
            return {"response": ""}
    except httpx.ConnectError as e:
        print(f"[ERROR] inference 连接失败: {e}, 重新创建连接...")
        client.close()
        client = get_client()
        return {"response": "", "error": str(e)}
    except Exception as e:
        print(f"[ERROR] inference 异常: {e}")
        return {"response": "", "error": str(e)}


def main():
    print("[INFO] 启动 bidder 服务...")
    register()

    print("[INFO] 开始轮询任务...")
    while True:
        try:
            overview = query_task()
            task_id = overview.get("task_id")
            if task_id is None:
                time.sleep(0.5)
                continue

            target_sla = overview.get("target_sla", "Gold")
            print(f"[INFO] 收到任务 {task_id}, target_sla={target_sla}")

            result = ask_task(task_id, target_sla)
            if result.get("status") != "accepted":
                print(f"[WARN] 任务 {task_id} 未被接受: {result}")
                continue

            task = result.get("task", {})
            print(f"[INFO] 开始推理任务 {task_id}...")

            for msg in task.get("messages", []):
                rt = msg.get("eval_request_type", "loglikelihood")
                prompt = msg.get("prompt", "")
                gen_kwargs = msg.get("eval_gen_kwargs", {})
                continuation = msg.get("eval_continuation", "")

                result = do_inference(prompt, rt, gen_kwargs, continuation)
                msg.update(result)

            submit_result = submit_task(task)
            print(f"[INFO] 任务 {task_id} 提交成功: {submit_result}")

            score_info = get_score()
            if score_info:
                print(f"[INFO] 当前分数: {score_info.get('score')}, 任务数: {score_info.get('tasks_completed')}, 排名: {score_info.get('rank')}")

        except KeyboardInterrupt:
            print("[INFO] 收到中断信号，退出...")
            break
        except Exception as e:
            print(f"[ERROR] 主循环异常: {e}")
            time.sleep(1)

    client.close()
    print("[INFO] bidder 服务已退出")


if __name__ == "__main__":
    main()
