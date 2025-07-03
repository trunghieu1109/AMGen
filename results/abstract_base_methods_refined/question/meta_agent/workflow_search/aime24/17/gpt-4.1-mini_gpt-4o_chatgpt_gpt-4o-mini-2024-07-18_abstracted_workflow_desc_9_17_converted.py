async def forward_17(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Rewrite the polynomial expression a^2b + a^2c + b^2a + b^2c + c^2a + c^2b in a symmetric form using variables a, b, c, and express it in terms of symmetric sums or simpler symmetric expressions to facilitate algebraic manipulation."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, rewrite polynomial symmetrically, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Express the symmetric polynomial from subtask_1 in terms of the elementary symmetric sums S1 = a + b + c, S2 = ab + bc + ca, and S3 = abc, and simplify the expression using the constraint S1 = 300."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, express polynomial in symmetric sums, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_reflect_instruction_3 = "Sub-task 3: Derive an explicit algebraic relation between S2 and S3 from the equation obtained in subtask_2, ensuring correctness by performing two independent derivations and cross-verifying results to avoid algebraic errors."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2, answer2], cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, first derivation of S2-S3 relation, thinking: {thinking3a.content}; answer: {answer3a.content}")
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking2, answer2], cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, second derivation of S2-S3 relation, thinking: {thinking3b.content}; answer: {answer3b.content}")
    feedback, correct = await critic_agent_3([taskInfo, thinking3a, answer3a, thinking3b, answer3b], "Review the two derivations of the relation between S2 and S3 and verify consistency.", 0, is_sub_task=True)
    agents.append(f"Critic agent {critic_agent_3.id}, feedback on derivations, thinking: {feedback.content}; answer: {correct.content}")
    iteration = 0
    max_iter = self.max_round
    while correct.content != "True" and iteration < max_iter:
        thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking2, answer2, thinking3a, answer3a, thinking3b, answer3b, feedback], cot_reflect_instruction_3, iteration + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3a.id}, refining first derivation, thinking: {thinking3a.content}; answer: {answer3a.content}")
        thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking2, answer2, thinking3a, answer3a, thinking3b, answer3b, feedback], cot_reflect_instruction_3, iteration + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining second derivation, thinking: {thinking3b.content}; answer: {answer3b.content}")
        feedback, correct = await critic_agent_3([taskInfo, thinking3a, answer3a, thinking3b, answer3b], "Review the two derivations of the relation between S2 and S3 and verify consistency.", iteration + 1, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback on derivations, thinking: {feedback.content}; answer: {correct.content}")
        iteration += 1
    final_answer3 = answer3a if correct.content == "True" else answer3a
    final_thinking3 = thinking3a if correct.content == "True" else thinking3a
    sub_tasks.append(f"Sub-task 3 output: thinking - {final_thinking3.content}; answer - {final_answer3.content}")
    subtask_desc3['response'] = {
        "thinking": final_thinking3,
        "answer": final_answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = "Sub-task 4: Translate the problem constraints into an equation involving a, b, c with a + b + c = 300 and the polynomial expression equal to 6,000,000, then reformulate it as an equation in two variables by substituting c = 300 - a - b."
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, final_thinking3, final_answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, reformulate constraints with c=300-a-b, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_5a = "Sub-task 5a: Analyze the case where one variable is zero (e.g., c=0), reduce the problem to two variables, and derive the resulting quadratic or polynomial equation to find all nonnegative integer solutions (a, b) satisfying the constraints."
    cot_sc_instruction_5b = "Sub-task 5b: Analyze the fully symmetric case where a = b = c, check if it satisfies the polynomial equation and constraints, and determine if it yields valid solutions."
    cot_sc_instruction_5c = "Sub-task 5c: Analyze cases where two variables are equal and the third is distinct (e.g., a = b ≠ c), derive the corresponding equations, and find all nonnegative integer solutions satisfying the constraints."
    cot_sc_instruction_5d = "Sub-task 5d: Analyze the general case where a, b, c are distinct nonnegative integers summing to 300, and use the polynomial equation to identify all valid triples, ensuring no solutions are missed by systematic exploration."

    cot_agents_5a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_5b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_5c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_5d = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]

    possible_answers_5a = []
    thinkingmapping_5a = {}
    answermapping_5a = {}
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_sc_instruction_5a,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5a, answer5a = await cot_agents_5a[i]([taskInfo, thinking4, answer4], cot_sc_instruction_5a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5a[i].id}, analyze c=0 case, thinking: {thinking5a.content}; answer: {answer5a.content}")
        possible_answers_5a.append(answer5a.content)
        thinkingmapping_5a[answer5a.content] = thinking5a
        answermapping_5a[answer5a.content] = answer5a
    answer5a_content = Counter(possible_answers_5a).most_common(1)[0][0]
    thinking5a = thinkingmapping_5a[answer5a_content]
    answer5a = answermapping_5a[answer5a_content]
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {
        "thinking": thinking5a,
        "answer": answer5a
    }
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])

    possible_answers_5b = []
    thinkingmapping_5b = {}
    answermapping_5b = {}
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_sc_instruction_5b,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5b, answer5b = await cot_agents_5b[i]([taskInfo, thinking4, answer4], cot_sc_instruction_5b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5b[i].id}, analyze symmetric case a=b=c, thinking: {thinking5b.content}; answer: {answer5b.content}")
        possible_answers_5b.append(answer5b.content)
        thinkingmapping_5b[answer5b.content] = thinking5b
        answermapping_5b[answer5b.content] = answer5b
    answer5b_content = Counter(possible_answers_5b).most_common(1)[0][0]
    thinking5b = thinkingmapping_5b[answer5b_content]
    answer5b = answermapping_5b[answer5b_content]
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {
        "thinking": thinking5b,
        "answer": answer5b
    }
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])

    possible_answers_5c = []
    thinkingmapping_5c = {}
    answermapping_5c = {}
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": cot_sc_instruction_5c,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5c, answer5c = await cot_agents_5c[i]([taskInfo, thinking4, answer4], cot_sc_instruction_5c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5c[i].id}, analyze case a=b!=c, thinking: {thinking5c.content}; answer: {answer5c.content}")
        possible_answers_5c.append(answer5c.content)
        thinkingmapping_5c[answer5c.content] = thinking5c
        answermapping_5c[answer5c.content] = answer5c
    answer5c_content = Counter(possible_answers_5c).most_common(1)[0][0]
    thinking5c = thinkingmapping_5c[answer5c_content]
    answer5c = answermapping_5c[answer5c_content]
    sub_tasks.append(f"Sub-task 5c output: thinking - {thinking5c.content}; answer - {answer5c.content}")
    subtask_desc5c['response'] = {
        "thinking": thinking5c,
        "answer": answer5c
    }
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])

    possible_answers_5d = []
    thinkingmapping_5d = {}
    answermapping_5d = {}
    subtask_desc5d = {
        "subtask_id": "subtask_5d",
        "instruction": cot_sc_instruction_5d,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5d, answer5d = await cot_agents_5d[i]([taskInfo, thinking4, answer4], cot_sc_instruction_5d, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5d[i].id}, analyze general distinct case, thinking: {thinking5d.content}; answer: {answer5d.content}")
        possible_answers_5d.append(answer5d.content)
        thinkingmapping_5d[answer5d.content] = thinking5d
        answermapping_5d[answer5d.content] = answer5d
    answer5d_content = Counter(possible_answers_5d).most_common(1)[0][0]
    thinking5d = thinkingmapping_5d[answer5d_content]
    answer5d = answermapping_5d[answer5d_content]
    sub_tasks.append(f"Sub-task 5d output: thinking - {thinking5d.content}; answer - {answer5d.content}")
    subtask_desc5d['response'] = {
        "thinking": thinking5d,
        "answer": answer5d
    }
    logs.append(subtask_desc5d)
    print("Step 5d: ", sub_tasks[-1])

    cot_instruction_6 = "Sub-task 6: Implement a systematic enumeration algorithm over all triples (a, b, c) of nonnegative integers with a + b + c = 300, using the algebraic conditions and simplifications from previous subtasks to prune the search space and efficiently identify all solutions satisfying the polynomial equation."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a", "thinking of subtask 5b", "answer of subtask 5b", "thinking of subtask 5c", "answer of subtask 5c", "thinking of subtask 5d", "answer of subtask 5d"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5a, answer5a, thinking5b, answer5b, thinking5c, answer5c, thinking5d, answer5d], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, enumerate all valid triples, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_7 = "Sub-task 7: Count the total number of valid triples (a, b, c) found in subtask_6 and return this count as the final numeric answer."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, count total valid triples, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
