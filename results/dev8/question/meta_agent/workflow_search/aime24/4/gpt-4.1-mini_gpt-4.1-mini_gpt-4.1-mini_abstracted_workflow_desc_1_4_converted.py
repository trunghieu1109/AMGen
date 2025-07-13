async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Identify candidate primes p such that the congruence n^4 ≡ -1 (mod p) has solutions. "
        "For each candidate prime p, enumerate all four roots n modulo p satisfying n^4 ≡ -1 (mod p). "
        "Output a JSON list of objects with keys 'p' and 'roots_mod_p' (list of roots modulo p). "
        "Do not assume solutions modulo p lift to p^2 without verification."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying candidate primes and roots modulo p, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2a = (
        "Sub-task 2a: For each candidate prime p and each root r modulo p from Sub-task 1, apply Hensel's lemma to lift r to roots modulo p^2. "
        "Enumerate all lifted roots of the form r + p*k for k in [0, p-1] that satisfy m^4 ≡ -1 (mod p^2). "
        "Output a JSON list of objects with keys 'p', 'root_mod_p', and 'lifted_roots_mod_p2' (list of lifted roots modulo p^2). "
        "Do not skip any roots or assume uniqueness of lifts."
    )
    N = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2a = []
    possible_thinkings_2a = []
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2a, answer2a = await cot_agents_2a[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, lifting roots modulo p to modulo p^2, thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a)
        possible_thinkings_2a.append(thinking2a)
    counter_2a = Counter([ans.content for ans in possible_answers_2a])
    most_common_answer_2a = counter_2a.most_common(1)[0][0]
    idx_2a = [ans.content for ans in possible_answers_2a].index(most_common_answer_2a)
    thinking2a_final = possible_thinkings_2a[idx_2a]
    answer2a_final = possible_answers_2a[idx_2a]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a_final.content}; answer - {answer2a_final.content}")
    subtask_desc2a['response'] = {"thinking": thinking2a_final, "answer": answer2a_final}
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])

    cot_sc_instruction_2b = (
        "Sub-task 2b: From the lifted roots modulo p^2 obtained in Sub-task 2a, enumerate all positive integers m with 1 ≤ m < p^2 such that m^4 ≡ -1 (mod p^2). "
        "Explicitly verify the congruence for each candidate. Identify and output the minimal such m for the least prime p found. "
        "Output JSON with keys 'p' and 'm'. Do not assume minimal root modulo p yields minimal m modulo p^2; check all candidates exhaustively."
    )
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2b = []
    possible_thinkings_2b = []
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", thinking2a_final.content, answer2a_final.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2b, answer2b = await cot_agents_2b[i]([taskInfo, thinking2a_final, answer2a_final], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, enumerating minimal m modulo p^2, thinking: {thinking2b.content}; answer: {answer2b.content}")
        possible_answers_2b.append(answer2b)
        possible_thinkings_2b.append(thinking2b)
    counter_2b = Counter([ans.content for ans in possible_answers_2b])
    most_common_answer_2b = counter_2b.most_common(1)[0][0]
    idx_2b = [ans.content for ans in possible_answers_2b].index(most_common_answer_2b)
    thinking2b_final = possible_thinkings_2b[idx_2b]
    answer2b_final = possible_answers_2b[idx_2b]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b_final.content}; answer - {answer2b_final.content}")
    subtask_desc2b['response'] = {"thinking": thinking2b_final, "answer": answer2b_final}
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])

    reflect_inst_3 = (
        "Sub-task 3: Validate and verify the final answers for least prime p and minimal positive integer m such that m^4 + 1 divisible by p^2. "
        "Confirm correctness of modular arithmetic, verify no smaller prime or integer satisfies the conditions, and provide a concise summary with rigorous justification. "
        "Explicitly confirm minimality without assumptions. Output JSON with keys 'p' and 'm'."
    )
    cot_reflect_instruction_3 = "Sub-task 3: Validate and verify final answers. " + reflect_inst_3
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2a_final, answer2a_final, thinking2b_final, answer2b_final]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", thinking1.content, answer1.content, thinking2a_final.content, answer2a_final.content, thinking2b_final.content, answer2b_final.content],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, validating final answers, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                               "Please review and provide the limitations of provided solutions. If absolutely correct, output exactly 'True' in 'correct'.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining final answers, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
