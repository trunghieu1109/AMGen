async def forward_19(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = "Sub-task 1a: Explicitly expand and simplify the expression 2 - 2ω^k + ω^{2k} for a general k, writing it in standard polynomial form in terms of ω^k without assuming any factorization."
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1a, answer1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, expand and simplify expression 2 - 2ω^k + ω^2k, thinking: {thinking1a.content}; answer: {answer1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc1a['response'] = {
        "thinking": thinking1a,
        "answer": answer1a
    }
    logs.append(subtask_desc1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_instruction_1b = "Sub-task 1b: Verify any proposed factorization of the expression 2 - 2ω^k + ω^{2k} step-by-step, checking if it can be expressed as a perfect square or another simpler polynomial form, and identify any constant term differences."
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "CoT"
    }
    thinking1b, answer1b = await cot_agent_1b([taskInfo, thinking1a, answer1a], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, verify factorization of expression, thinking: {thinking1b.content}; answer: {answer1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc1b['response'] = {
        "thinking": thinking1b,
        "answer": answer1b
    }
    logs.append(subtask_desc1b)
    print("Step 1b: ", sub_tasks[-1])
    
    cot_instruction_1c = "Sub-task 1c: Validate the expression 2 - 2ω^k + ω^{2k} and any factorization by substituting specific values of k (e.g., k=0) or ω^k (e.g., ω^0=1) to check consistency and correctness of the algebraic form."
    cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_instruction_1c,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "CoT"
    }
    thinking1c, answer1c = await cot_agent_1c([taskInfo, thinking1a, answer1a, thinking1b, answer1b], cot_instruction_1c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1c.id}, validate expression by substitution, thinking: {thinking1c.content}; answer: {answer1c.content}")
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking1c.content}; answer - {answer1c.content}")
    subtask_desc1c['response'] = {
        "thinking": thinking1c,
        "answer": answer1c
    }
    logs.append(subtask_desc1c)
    print("Step 1c: ", sub_tasks[-1])
    
    reflexion_instruction_2a = "Sub-task 2a: Conduct a Reflexion and Debate process where multiple agents independently expand and factor the expression 2 - 2ω^k + ω^{2k}, then compare results term-by-term to detect and resolve any discrepancies or errors in factorization."
    cot_agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2a = self.max_round
    cot_inputs_2a = [taskInfo, thinking1a, answer1a, thinking1b, answer1b, thinking1c, answer1c]
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": reflexion_instruction_2a,
        "context": ["user query", "thinking and answer of subtasks 1a, 1b, 1c"],
        "agent_collaboration": "Reflexion"
    }
    thinking2a, answer2a = await cot_agent_2a(cot_inputs_2a, reflexion_instruction_2a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2a.id}, initial expansion and factorization, thinking: {thinking2a.content}; answer: {answer2a.content}")
    for i in range(N_max_2a):
        feedback, correct = await critic_agent_2a([taskInfo, thinking2a, answer2a], "Please review the factorization and provide feedback on correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2a.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_2a.extend([thinking2a, answer2a, feedback])
        thinking2a, answer2a = await cot_agent_2a(cot_inputs_2a, reflexion_instruction_2a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2a.id}, refinement round {i+1}, thinking: {thinking2a.content}; answer: {answer2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {
        "thinking": thinking2a,
        "answer": answer2a
    }
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])
    
    cot_sc_instruction_2b = "Sub-task 2b: Based on the verified and agreed algebraic form from subtask_2a, define the polynomial P(x) = 2 - 2x + x^2 and express the product ∏_{k=0}^{12} P(ω^k) in terms of roots of unity and related polynomial identities."
    N = self.max_sc
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", "thinking and answer of subtask 2a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2b, answer2b = await cot_agents_2b[i]([taskInfo, thinking2a, answer2a], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, define polynomial and express product over roots, thinking: {thinking2b.content}; answer: {answer2b.content}")
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
    
    cot_instruction_3a = "Sub-task 3a: Use the factorization of x^{13} - 1 = ∏_{k=0}^{12} (x - ω^k) and properties of roots of unity to rewrite the product ∏_{k=0}^{12} P(ω^k) as a resultant or polynomial evaluation involving P(x) and x^{13} - 1."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking and answer of subtask 2b"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2b, answer2b], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, rewrite product using roots of unity properties, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    
    cot_sc_instruction_3b = "Sub-task 3b: Calculate the exact value of the product ∏_{k=0}^{12} P(ω^k) by evaluating the polynomial expressions and simplifying using algebraic identities related to roots of unity."
    cot_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3b = []
    thinkingmapping_3b = {}
    answermapping_3b = {}
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_sc_instruction_3b,
        "context": ["user query", "thinking and answer of subtask 3a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3b, answer3b = await cot_agents_3b[i]([taskInfo, thinking3a, answer3a], cot_sc_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3b[i].id}, calculate exact product value, thinking: {thinking3b.content}; answer: {answer3b.content}")
        possible_answers_3b.append(answer3b.content)
        thinkingmapping_3b[answer3b.content] = thinking3b
        answermapping_3b[answer3b.content] = answer3b
    answer3b_content = Counter(possible_answers_3b).most_common(1)[0][0]
    thinking3b = thinkingmapping_3b[answer3b_content]
    answer3b = answermapping_3b[answer3b_content]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    
    cot_instruction_4a = "Sub-task 4a: Find the remainder when the integer obtained from subtask_3b is divided by 1000."
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_instruction_4a,
        "context": ["user query", "thinking and answer of subtask 3b"],
        "agent_collaboration": "CoT"
    }
    thinking4a, answer4a = await cot_agent_4a([taskInfo, thinking3b, answer3b], cot_instruction_4a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4a.id}, compute remainder modulo 1000, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {
        "thinking": thinking4a,
        "answer": answer4a
    }
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    
    reflexion_instruction_5a = "Sub-task 5a: Perform a Reflexion checkpoint reviewing all algebraic transformations, substitutions, and calculations from stages 1 to 4 to ensure no algebraic or logical errors remain before finalizing the answer."
    cot_agent_5a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5a = self.max_round
    cot_inputs_5a = [taskInfo, thinking1a, answer1a, thinking1b, answer1b, thinking1c, answer1c, thinking2a, answer2a, thinking2b, answer2b, thinking3a, answer3a, thinking3b, answer3b, thinking4a, answer4a]
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": reflexion_instruction_5a,
        "context": ["user query", "all previous thinking and answers"],
        "agent_collaboration": "Reflexion"
    }
    thinking5a, answer5a = await cot_agent_5a(cot_inputs_5a, reflexion_instruction_5a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5a.id}, initial review, thinking: {thinking5a.content}; answer: {answer5a.content}")
    for i in range(N_max_5a):
        feedback, correct = await critic_agent_5a([taskInfo, thinking5a, answer5a], "Please review the entire reasoning chain for errors and provide feedback.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5a.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_5a.extend([thinking5a, answer5a, feedback])
        thinking5a, answer5a = await cot_agent_5a(cot_inputs_5a, reflexion_instruction_5a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5a.id}, refinement round {i+1}, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {
        "thinking": thinking5a,
        "answer": answer5a
    }
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5a, answer5a, sub_tasks, agents)
    return final_answer, logs
