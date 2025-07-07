async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Understand the problem statement and clarify the conditions: identify that p is the least prime such that there exists a positive integer n with p^2 dividing n^4 + 1, and that we need to find the least positive integer m such that m^4 + 1 is divisible by p^2."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, understanding problem statement, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2a = "Sub-task 2a: Determine the necessary condition on prime p by analyzing the congruence n^4 ≡ -1 (mod p). Identify primes p for which -1 is a quartic residue modulo p, focusing on primes p ≡ 1 (mod 8)."
    N = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2a, answer2a = await cot_agents_2a[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, determining primes p with n^4 ≡ -1 (mod p), thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a.content)
        thinkingmapping_2a[answer2a.content] = thinking2a
        answermapping_2a[answer2a.content] = answer2a
    answer2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking2a = thinkingmapping_2a[answer2a_content]
    answer2a = answermapping_2a[answer2a_content]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {
        "thinking": thinking2a,
        "answer": answer2a
    }
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])
    
    cot_sc_instruction_2b = "Sub-task 2b: For each candidate prime p from subtask_2a, find all n modulo p such that n^4 ≡ -1 (mod p)."
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2b, answer2b = await cot_agents_2b[i]([taskInfo, thinking1, answer1, thinking2a, answer2a], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, finding all n modulo p with n^4 ≡ -1 (mod p), thinking: {thinking2b.content}; answer: {answer2b.content}")
        possible_answers_2b.append(answer2b.content)
        thinkingmapping_2b[answer2b.content] = thinking2b
        answermapping_2b[answer2b.content] = answer2b
    answer2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking2b = thinkingmapping_2b[answer2b_content]
    answer2b = answermapping_2b[answer2b_content]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {
        "thinking": thinking2b,
        "answer": answer2b
    }
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])
    
    cot_sc_instruction_2c = "Sub-task 2c: For each pair (p, n) found in subtask_2b, attempt to lift the solution n modulo p to a solution modulo p^2 by solving n^4 ≡ -1 (mod p^2), using Hensel's lemma or systematic trial."
    cot_agents_2c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2c = []
    thinkingmapping_2c = {}
    answermapping_2c = {}
    subtask_desc2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_sc_instruction_2c,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2a", "answer of subtask 2a", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2c, answer2c = await cot_agents_2c[i]([taskInfo, thinking1, answer1, thinking2a, answer2a, thinking2b, answer2b], cot_sc_instruction_2c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2c[i].id}, lifting solutions modulo p to modulo p^2, thinking: {thinking2c.content}; answer: {answer2c.content}")
        possible_answers_2c.append(answer2c.content)
        thinkingmapping_2c[answer2c.content] = thinking2c
        answermapping_2c[answer2c.content] = answer2c
    answer2c_content = Counter(possible_answers_2c).most_common(1)[0][0]
    thinking2c = thinkingmapping_2c[answer2c_content]
    answer2c = answermapping_2c[answer2c_content]
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking2c.content}; answer - {answer2c.content}")
    subtask_desc2c['response'] = {
        "thinking": thinking2c,
        "answer": answer2c
    }
    logs.append(subtask_desc2c)
    print("Step 2c: ", sub_tasks[-1])
    
    cot_sc_instruction_2d = "Sub-task 2d: Verify for each candidate (p, n) from subtask_2c that p^2 divides n^4 + 1 by explicit computation to avoid arithmetic errors. Include sanity checks for each divisibility claim."
    cot_agents_2d = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2d = []
    thinkingmapping_2d = {}
    answermapping_2d = {}
    subtask_desc2d = {
        "subtask_id": "subtask_2d",
        "instruction": cot_sc_instruction_2d,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2a", "answer of subtask 2a", "thinking of subtask 2b", "answer of subtask 2b", "thinking of subtask 2c", "answer of subtask 2c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2d, answer2d = await cot_agents_2d[i]([taskInfo, thinking1, answer1, thinking2a, answer2a, thinking2b, answer2b, thinking2c, answer2c], cot_sc_instruction_2d, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2d[i].id}, verifying divisibility p^2 | n^4 + 1 with sanity checks, thinking: {thinking2d.content}; answer: {answer2d.content}")
        possible_answers_2d.append(answer2d.content)
        thinkingmapping_2d[answer2d.content] = thinking2d
        answermapping_2d[answer2d.content] = answer2d
    answer2d_content = Counter(possible_answers_2d).most_common(1)[0][0]
    thinking2d = thinkingmapping_2d[answer2d_content]
    answer2d = answermapping_2d[answer2d_content]
    sub_tasks.append(f"Sub-task 2d output: thinking - {thinking2d.content}; answer - {answer2d.content}")
    subtask_desc2d['response'] = {
        "thinking": thinking2d,
        "answer": answer2d
    }
    logs.append(subtask_desc2d)
    print("Step 2d: ", sub_tasks[-1])
    
    cot_reflect_instruction_3 = "Sub-task 3: Identify the least prime p for which there exists a positive integer n such that p^2 divides n^4 + 1, based on verified candidates from subtask_2d. Use Reflexion pattern to refine and confirm the least prime."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2a, answer2a, thinking2b, answer2b, thinking2c, answer2c, thinking2d, answer2d]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2a", "answer of subtask 2a", "thinking of subtask 2b", "answer of subtask 2b", "thinking of subtask 2c", "answer of subtask 2c", "thinking of subtask 2d", "answer of subtask 2d"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, identifying least prime p, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the identification of least prime p and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining least prime p identification, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_sc_instruction_4a = "Sub-task 4a: Given the prime p from subtask_3, systematically check all positive integers m from 1 up to p^2 - 1 to find all m such that m^4 ≡ -1 (mod p^2). Agents must check all candidates thoroughly."
    cot_agents_4a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4a = []
    thinkingmapping_4a = {}
    answermapping_4a = {}
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_sc_instruction_4a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4a, answer4a = await cot_agents_4a[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4a[i].id}, checking all m from 1 to p^2-1 for m^4 ≡ -1 (mod p^2), thinking: {thinking4a.content}; answer: {answer4a.content}")
        possible_answers_4a.append(answer4a.content)
        thinkingmapping_4a[answer4a.content] = thinking4a
        answermapping_4a[answer4a.content] = answer4a
    answer4a_content = Counter(possible_answers_4a).most_common(1)[0][0]
    thinking4a = thinkingmapping_4a[answer4a_content]
    answer4a = answermapping_4a[answer4a_content]
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {
        "thinking": thinking4a,
        "answer": answer4a
    }
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    
    cot_sc_instruction_4b = "Sub-task 4b: For each candidate m found in subtask_4a, verify explicitly that p^2 divides m^4 + 1 to avoid arithmetic errors. Include sanity checks for each verification."
    cot_agents_4b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4b = []
    thinkingmapping_4b = {}
    answermapping_4b = {}
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_sc_instruction_4b,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4b, answer4b = await cot_agents_4b[i]([taskInfo, thinking3, answer3, thinking4a, answer4a], cot_sc_instruction_4b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4b[i].id}, verifying divisibility p^2 | m^4 + 1 for candidates, thinking: {thinking4b.content}; answer: {answer4b.content}")
        possible_answers_4b.append(answer4b.content)
        thinkingmapping_4b[answer4b.content] = thinking4b
        answermapping_4b[answer4b.content] = answer4b
    answer4b_content = Counter(possible_answers_4b).most_common(1)[0][0]
    thinking4b = thinkingmapping_4b[answer4b_content]
    answer4b = answermapping_4b[answer4b_content]
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {
        "thinking": thinking4b,
        "answer": answer4b
    }
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    
    cot_sc_instruction_4c = "Sub-task 4c: From the verified candidates in subtask_4b, determine the least positive integer m such that m^4 + 1 is divisible by p^2. Use Self-Consistency Chain-of-Thought to ensure thoroughness and correctness."
    cot_agents_4c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4c = []
    thinkingmapping_4c = {}
    answermapping_4c = {}
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_sc_instruction_4c,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4a", "answer of subtask 4a", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4c, answer4c = await cot_agents_4c[i]([taskInfo, thinking3, answer3, thinking4a, answer4a, thinking4b, answer4b], cot_sc_instruction_4c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4c[i].id}, determining least positive integer m, thinking: {thinking4c.content}; answer: {answer4c.content}")
        possible_answers_4c.append(answer4c.content)
        thinkingmapping_4c[answer4c.content] = thinking4c
        answermapping_4c[answer4c.content] = answer4c
    answer4c_content = Counter(possible_answers_4c).most_common(1)[0][0]
    thinking4c = thinkingmapping_4c[answer4c_content]
    answer4c = answermapping_4c[answer4c_content]
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {
        "thinking": thinking4c,
        "answer": answer4c
    }
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking4c, answer4c, sub_tasks, agents)
    return final_answer, logs
