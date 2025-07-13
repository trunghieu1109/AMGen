async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 0: Analyze the divisibility condition n^4 + 1 ≡ 0 (mod p^2) to find the least prime p for which there exists a positive integer n satisfying p^2 | n^4 + 1. "
        "Steps: (a) Identify primes p for which x^4 ≡ -1 (mod p) has solutions; (b) For each such prime, check if solutions lift to modulo p^2 by evaluating f'(x) = 4x^3 modulo p and p^2, including gcd(f'(x_0), p) to determine applicability of Hensel's lemma; (c) Document all modular arithmetic, gcd checks, and derivative invertibility; (d) Identify the least prime p with a solution modulo p^2 and provide at least one such n modulo p^2. "
        "Explicitly handle cases where derivative is non-invertible and provide warnings about lifting limitations."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_0",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, analyzing divisibility condition, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {
        "thinking": thinking_0,
        "answer": answer_0
    }
    logs.append(subtask_desc_0)

    cot_instruction_1a = (
        "Sub-task 1a: Given the least prime p and a solution n modulo p^2 from Sub-task 0, enumerate all candidate positive integers m modulo p^2 satisfying m^4 ≡ -1 (mod p^2). "
        "Include: (a) generating candidates by lifting solutions modulo p to modulo p^2, (b) handling non-invertible derivative cases by alternative lifting or direct enumeration, (c) restricting search space using congruence classes modulo p and small positive integers, (d) output a concise candidate list for verification. "
        "Use Chain-of-Thought and Reflexion to ensure completeness and correctness of candidate enumeration."
    )
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query", "thinking of subtask 0", "answer of subtask 0"],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo, thinking_0, answer_0], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, enumerating candidate m values, thinking: {thinking_1a.content}; answer: {answer_1a.content}")

    cot_inputs_1a = [taskInfo, thinking_0, answer_0, thinking_1a, answer_1a]
    for i in range(self.max_round):
        feedback_1a, correct_1a = await critic_agent_1a(cot_inputs_1a, "Please review and critique the candidate enumeration for completeness and correctness. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1a.id}, feedback: {feedback_1a.content}; correctness: {correct_1a.content}")
        if correct_1a.content == "True":
            break
        cot_inputs_1a.extend([thinking_1a, answer_1a, feedback_1a])
        thinking_1a, answer_1a = await cot_agent_1a(cot_inputs_1a, cot_instruction_1a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1a.id}, refining candidate enumeration, thinking: {thinking_1a.content}; answer: {answer_1a.content}")

    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {
        "thinking": thinking_1a,
        "answer": answer_1a
    }
    logs.append(subtask_desc_1a)

    cot_instruction_1b = (
        "Sub-task 1b: Verify each candidate m from Sub-task 1a by explicitly computing m^4 + 1 modulo p^2 and confirming divisibility by p^2. "
        "Retain only candidates passing verification and select the least positive integer m. Include detailed modular arithmetic and verification steps. "
        "If no candidates pass, trigger fallback strategies such as extending search or revisiting lifting assumptions. Use Chain-of-Thought and Debate to ensure correctness and consensus."
    )
    cot_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    debate_agents_1b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask 0", "answer of subtask 0", "thinking of subtask 1a", "answer of subtask 1a"],
        "agent_collaboration": "CoT | Debate"
    }

    possible_answers_1b = []
    possible_thinkings_1b = []
    for i in range(self.max_sc):
        thinking_1b, answer_1b = await cot_agents_1b[i]([taskInfo, thinking_0, answer_0, thinking_1a, answer_1a], cot_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1b[i].id}, verifying candidates, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
        possible_answers_1b.append(answer_1b)
        possible_thinkings_1b.append(thinking_1b)

    all_thinking_1b = [[] for _ in range(self.max_round)]
    all_answer_1b = [[] for _ in range(self.max_round)]

    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_1b):
            if r == 0:
                thinking_d, answer_d = await agent([taskInfo, thinking_0, answer_0, thinking_1a, answer_1a] + possible_thinkings_1b + possible_answers_1b, cot_instruction_1b, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_0, answer_0, thinking_1a, answer_1a] + all_thinking_1b[r-1] + all_answer_1b[r-1]
                thinking_d, answer_d = await agent(input_infos, cot_instruction_1b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying candidates, thinking: {thinking_d.content}; answer: {answer_d.content}")
            all_thinking_1b[r].append(thinking_d)
            all_answer_1b[r].append(answer_d)

    final_decision_agent_1b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1b_final, answer_1b_final = await final_decision_agent_1b([taskInfo] + all_thinking_1b[-1] + all_answer_1b[-1], "Sub-task 1b: Synthesize and select the minimal verified m satisfying m^4 + 1 ≡ 0 (mod p^2).", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting minimal verified m, thinking: {thinking_1b_final.content}; answer: {answer_1b_final.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b_final.content}; answer - {answer_1b_final.content}")
    subtask_desc_1b['response'] = {
        "thinking": thinking_1b_final,
        "answer": answer_1b_final
    }
    logs.append(subtask_desc_1b)

    debate_instruction_2 = (
        "Sub-task 2: Consolidate and validate the findings: (a) confirm that the identified prime p is the smallest prime with a solution n such that p^2 divides n^4 + 1, "
        "(b) confirm that the minimal m found satisfies m^4 + 1 ≡ 0 (mod p^2) with explicit modular arithmetic verification, "
        "(c) provide a clear, formal statement of the solution including values of p and m, and (d) include proof sketches or reasoning about uniqueness or minimality. "
        "Use Debate and Reflexion to resolve any contradictions or uncertainties before finalizing."
    )
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    reflexion_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", "thinking of subtask 0", "answer of subtask 0", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "Debate | Reflexion"
    }

    all_thinking_2 = [[] for _ in range(self.max_round)]
    all_answer_2 = [[] for _ in range(self.max_round)]

    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking_2, answer_2 = await agent([taskInfo, thinking_0, answer_0, thinking_1b_final, answer_1b_final], debate_instruction_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking_0, answer_0, thinking_1b_final, answer_1b_final] + all_thinking_2[r-1] + all_answer_2[r-1]
                thinking_2, answer_2 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, consolidating and validating, thinking: {thinking_2.content}; answer: {answer_2.content}")
            all_thinking_2[r].append(thinking_2)
            all_answer_2[r].append(answer_2)

        reflexion_thinking, reflexion_answer = await reflexion_agent_2([taskInfo, thinking_0, answer_0, thinking_1b_final, answer_1b_final] + all_thinking_2[r] + all_answer_2[r], debate_instruction_2, r, is_sub_task=True)
        agents.append(f"Reflexion agent, round {r}, refining consolidation, thinking: {reflexion_thinking.content}; answer: {reflexion_answer.content}")

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_final, answer_2_final = await final_decision_agent_2([taskInfo, thinking_0, answer_0, thinking_1b_final, answer_1b_final, reflexion_thinking, reflexion_answer], "Sub-task 2: Finalize and confirm the minimal prime p and minimal m satisfying the problem conditions with full verification.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing solution, thinking: {thinking_2_final.content}; answer: {answer_2_final.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_final.content}; answer - {answer_2_final.content}")
    subtask_desc_2['response'] = {
        "thinking": thinking_2_final,
        "answer": answer_2_final
    }
    logs.append(subtask_desc_2)

    final_answer = await self.make_final_answer(thinking_2_final, answer_2_final, sub_tasks, agents)
    for i, step in enumerate(sub_tasks, 1):
        print(f"Step {i}: ", step)
    return final_answer, logs
