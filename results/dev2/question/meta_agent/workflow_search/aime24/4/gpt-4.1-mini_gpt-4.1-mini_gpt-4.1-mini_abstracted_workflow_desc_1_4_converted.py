async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = (
        "Sub-task 1: Analyze the divisibility condition n^4 + 1 divisible by p^2. "
        "Determine necessary properties of prime p and integer n such that p^2 divides n^4 + 1. "
        "Characterize such primes p and possible n."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing divisibility condition, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the characterization from Sub-task 1, identify the least prime p for which there exists a positive integer n "
        "such that p^2 divides n^4 + 1. Test primes and apply number-theoretic criteria derived previously."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, identifying least prime p, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3a = (
        "Sub-task 3a: Given the prime p found in Sub-task 2, explicitly compute all solutions n_0 modulo p to the congruence n^4 ≡ -1 (mod p). "
        "Enumerate and verify the complete set of roots, ensuring none are omitted."
    )
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2, answer2], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, computing all roots modulo p, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])

    cot_reflect_instruction_3b = (
        "Sub-task 3b: For each root n_0 found in Sub-task 3a, apply Hensel's lemma to lift n_0 to a root modulo p^2 of the congruence n^4 ≡ -1 (mod p^2). "
        "Compute all possible lifts and verify each candidate rigorously. Use reflexion and debate to cross-check correctness and reconcile contradictions."
    )
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    debate_agents_3b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3b = self.max_round
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_reflect_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT | Reflexion | Debate"
    }
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking3a, answer3a, thinking2, answer2], cot_reflect_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, lifting roots modulo p^2, thinking: {thinking3b.content}; answer: {answer3b.content}")
    for i in range(N_max_3b):
        feedback, correct = await critic_agent_3b([taskInfo, thinking3b, answer3b], "please review the lifting and verification of roots modulo p^2 and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs = [taskInfo, thinking3a, answer3a, thinking2, answer2, thinking3b, answer3b, feedback]
        thinking3b, answer3b = await cot_agent_3b(cot_inputs, cot_reflect_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining lifted roots, thinking: {thinking3b.content}; answer: {answer3b.content}")
    for r in range(N_max_3b):
        for i, agent in enumerate(debate_agents_3b):
            if r == 0:
                thinking_d, answer_d = await agent([taskInfo, thinking3a, answer3a, thinking3b, answer3b], cot_reflect_instruction_3b, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking3a, answer3a, thinking3b, answer3b] + all_thinking_d[r-1] + all_answer_d[r-1]
                thinking_d, answer_d = await agent(input_infos, cot_reflect_instruction_3b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating lifted roots, thinking: {thinking_d.content}; answer: {answer_d.content}")
            if r == 0:
                all_thinking_d = [[] for _ in range(N_max_3b)]
                all_answer_d = [[] for _ in range(N_max_3b)]
            all_thinking_d[r].append(thinking_d)
            all_answer_d[r].append(answer_d)
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])

    cot_instruction_3c = (
        "Sub-task 3c: From the lifted candidates modulo p^2 in Sub-task 3b, select the smallest positive integer m such that m^4 + 1 is divisible by p^2. "
        "Prepare this candidate for final verification."
    )
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_instruction_3c,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "CoT"
    }
    thinking3c, answer3c = await cot_agent_3c([taskInfo, thinking3b, answer3b], cot_instruction_3c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3c.id}, selecting smallest valid m, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {"thinking": thinking3c, "answer": answer3c}
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])

    cot_sc_instruction_4a = (
        "Sub-task 4a: Verify explicitly that the candidate m from Sub-task 3c satisfies m^4 + 1 ≡ 0 (mod p^2). "
        "Perform modular arithmetic checks and confirm divisibility. If verification fails, document the failure and trigger fallback procedures."
    )
    cot_agents_4a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4a = []
    thinkingmapping_4a = {}
    answermapping_4a = {}
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_sc_instruction_4a,
        "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "SC_CoT | Reflexion | Debate"
    }
    for i in range(N):
        thinking4a, answer4a = await cot_agents_4a[i]([taskInfo, thinking3c, answer3c, thinking3b, answer3b], cot_sc_instruction_4a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4a[i].id}, verifying candidate m divisibility, thinking: {thinking4a.content}; answer: {answer4a.content}")
        possible_answers_4a.append(answer4a.content)
        thinkingmapping_4a[answer4a.content] = thinking4a
        answermapping_4a[answer4a.content] = answer4a
    answer4a_content = Counter(possible_answers_4a).most_common(1)[0][0]
    thinking4a = thinkingmapping_4a[answer4a_content]
    answer4a = answermapping_4a[answer4a_content]
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])

    if "verification failed" in answer4a_content.lower() or "no" in answer4a_content.lower():
        cot_sc_instruction_4b = (
            "Sub-task 4b: Since verification in Sub-task 4a failed, perform a systematic search over all candidates m = n_0 + p*k for k in [0, p-1] "
            "and all roots n_0 modulo p to find a valid m modulo p^2 such that m^4 + 1 ≡ 0 (mod p^2). Verify each candidate rigorously and select the least positive integer solution. "
            "Use reflexion and debate to ensure correctness."
        )
        cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
        critic_agent_4b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
        debate_agents_4b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
        N_max_4b = self.max_round
        subtask_desc4b = {
            "subtask_id": "subtask_4b",
            "instruction": cot_sc_instruction_4b,
            "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a", "thinking of subtask 3a", "answer of subtask 3a"],
            "agent_collaboration": "SC_CoT | Reflexion | Debate"
        }
        thinking4b, answer4b = await cot_agent_4b([taskInfo, thinking4a, answer4a, thinking3a, answer3a], cot_sc_instruction_4b, 0, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, systematic search for valid m, thinking: {thinking4b.content}; answer: {answer4b.content}")
        for i in range(N_max_4b):
            feedback, correct = await critic_agent_4b([taskInfo, thinking4b, answer4b], "please review the systematic search and verification of candidates and provide limitations.", i, is_sub_task=True)
            agents.append(f"Critic agent {critic_agent_4b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
            if correct.content.strip().lower() == "true":
                break
            cot_inputs = [taskInfo, thinking4a, answer4a, thinking3a, answer3a, thinking4b, answer4b, feedback]
            thinking4b, answer4b = await cot_agent_4b(cot_inputs, cot_sc_instruction_4b, i + 1, is_sub_task=True)
            agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, refining search results, thinking: {thinking4b.content}; answer: {answer4b.content}")
        for r in range(N_max_4b):
            for i, agent in enumerate(debate_agents_4b):
                if r == 0:
                    thinking_d, answer_d = await agent([taskInfo, thinking4a, answer4a, thinking4b, answer4b], cot_sc_instruction_4b, r, is_sub_task=True)
                else:
                    input_infos = [taskInfo, thinking4a, answer4a, thinking4b, answer4b] + all_thinking_4b[r-1] + all_answer_4b[r-1]
                    thinking_d, answer_d = await agent(input_infos, cot_sc_instruction_4b, r, is_sub_task=True)
                agents.append(f"Debate agent {agent.id}, round {r}, debating systematic search, thinking: {thinking_d.content}; answer: {answer_d.content}")
                if r == 0:
                    all_thinking_4b = [[] for _ in range(N_max_4b)]
                    all_answer_4b = [[] for _ in range(N_max_4b)]
                all_thinking_4b[r].append(thinking_d)
                all_answer_4b[r].append(answer_d)
        sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
        subtask_desc4b['response'] = {"thinking": thinking4b, "answer": answer4b}
        logs.append(subtask_desc4b)
        print("Step 4b: ", sub_tasks[-1])
    else:
        thinking4b = None
        answer4b = None

    cot_sc_instruction_4c = (
        "Sub-task 4c: Provide the final answer m, the least positive integer such that m^4 + 1 is divisible by p^2, "
        "along with a comprehensive verification report confirming correctness and divisibility. Reconcile any contradictions found during verification."
    )
    cot_agents_4c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4c = []
    thinkingmapping_4c = {}
    answermapping_4c = {}
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_sc_instruction_4c,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "SC_CoT | Reflexion"
    }
    for i in range(N):
        inputs = [taskInfo, thinking4a, answer4a]
        if thinking4b and answer4b:
            inputs += [thinking4b, answer4b]
        thinking4c, answer4c = await cot_agents_4c[i](inputs, cot_sc_instruction_4c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4c[i].id}, finalizing answer and verification, thinking: {thinking4c.content}; answer: {answer4c.content}")
        possible_answers_4c.append(answer4c.content)
        thinkingmapping_4c[answer4c.content] = thinking4c
        answermapping_4c[answer4c.content] = answer4c
    answer4c_content = Counter(possible_answers_4c).most_common(1)[0][0]
    thinking4c = thinkingmapping_4c[answer4c_content]
    answer4c = answermapping_4c[answer4c_content]
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {"thinking": thinking4c, "answer": answer4c}
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4c, answer4c, sub_tasks, agents)
    return final_answer, logs
