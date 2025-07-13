async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Derive parametric representations for points A, B, C, and D on the hyperbola x^2/20 - y^2/24 = 1, "
        "incorporating the condition that the diagonals of rhombus ABCD intersect at the origin. Explicitly define parameters s and t for points B and D, "
        "and express points A and C accordingly."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, deriving parametric representations, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_reflect_instruction_2 = (
        "Sub-task 2: Validate that the parametric points A, B, C, and D satisfy the hyperbola equation and the rhombus property that the diagonals intersect at the origin. "
        "Confirm the orientation and sign conventions of parameters s and t, ensuring one sinh term is negative to correctly represent the rhombus configuration."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    cot_inputs_2 = [taskInfo, thinking1, answer1]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_reflect_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, validating parametric points and sign conventions, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(N_max_2):
        feedback, correct = await critic_agent_2([taskInfo, thinking2, answer2],
                                                "Please review the validation of parametric points and sign conventions for correctness and completeness.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining validation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Express the diagonals AC and BD in terms of parameters s and t, and derive formulas for their squared lengths, especially focusing on BD^2. "
        "Include intermediate expressions for hyperbolic functions and their relations."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, expressing diagonals and squared lengths, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4a = (
        "Sub-task 4a: Derive the exact algebraic condition for the equality of the rhombus side lengths using the parametric forms and hyperbola constants a^2=20 and b^2=24. "
        "Explicitly incorporate sign conventions for s and t, and obtain the relation involving hyperbolic functions before simplification."
    )
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_instruction_4a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4a, answer4a = await cot_agent_4a([taskInfo, thinking3, answer3], cot_instruction_4a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4a.id}, deriving algebraic side-length equality, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])

    debate_instruction_4b = (
        "Sub-task 4b: Solve the algebraic condition from Subtask 4a to find the precise relation between parameters s and t, "
        "verifying the correctness of the relation coth s coth t = 6/5. Use a Debate collaboration pattern where one agent derives the relation and another verifies the algebraic steps and sign conventions to catch errors."
    )
    debate_agents_4b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4b = self.max_round
    all_thinking4b = [[] for _ in range(N_max_4b)]
    all_answer4b = [[] for _ in range(N_max_4b)]
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": debate_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4b):
        for i, agent in enumerate(debate_agents_4b):
            if r == 0:
                thinking4b, answer4b = await agent([taskInfo, thinking4a, answer4a], debate_instruction_4b, r, is_sub_task=True)
            else:
                input_infos_4b = [taskInfo, thinking4a, answer4a] + all_thinking4b[r-1] + all_answer4b[r-1]
                thinking4b, answer4b = await agent(input_infos_4b, debate_instruction_4b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying algebraic relation, thinking: {thinking4b.content}; answer: {answer4b.content}")
            all_thinking4b[r].append(thinking4b)
            all_answer4b[r].append(answer4b)
    final_decision_agent_4b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4b, answer4b = await final_decision_agent_4b([taskInfo] + all_thinking4b[-1] + all_answer4b[-1], "Sub-task 4b: Make final decision on algebraic relation correctness.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding algebraic relation correctness, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {"thinking": thinking4b, "answer": answer4b}
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])

    cot_instruction_5a = (
        "Sub-task 5a: Express BD^2 fully as a function of a single parameter t by substituting the relation coth s coth t = 6/5 derived in Subtask 4b. "
        "Simplify the expression to a single-variable function suitable for analysis."
    )
    cot_agents_5a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_5a = []
    thinkingmapping_5a = {}
    answermapping_5a = {}
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_instruction_5a,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking5a, answer5a = await cot_agents_5a[i]([taskInfo, thinking4b, answer4b], cot_instruction_5a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5a[i].id}, expressing BD^2 as function of t, thinking: {thinking5a.content}; answer: {answer5a.content}")
        possible_answers_5a.append(answer5a.content)
        thinkingmapping_5a[answer5a.content] = thinking5a
        answermapping_5a[answer5a.content] = answer5a
    answer5a_content = Counter(possible_answers_5a).most_common(1)[0][0]
    thinking5a = thinkingmapping_5a[answer5a_content]
    answer5a = answermapping_5a[answer5a_content]
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {"thinking": thinking5a, "answer": answer5a}
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])

    cot_reflect_instruction_5b = (
        "Sub-task 5b: Analyze the domain and constraints on the parameter t imposed by the rhombus condition and hyperbola properties. "
        "Explicitly determine the valid range of t and corresponding s values, considering sign conventions and ensuring the parameters correspond to points on the hyperbola."
    )
    cot_agent_5b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5b = self.max_round
    cot_inputs_5b = [taskInfo, thinking5a, answer5a]
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_reflect_instruction_5b,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a"],
        "agent_collaboration": "Reflexion"
    }
    thinking5b, answer5b = await cot_agent_5b(cot_inputs_5b, cot_reflect_instruction_5b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5b.id}, analyzing domain and constraints of t, thinking: {thinking5b.content}; answer: {answer5b.content}")
    for i in range(N_max_5b):
        feedback, correct = await critic_agent_5b([taskInfo, thinking5b, answer5b],
                                                "Please review the domain analysis and constraints on parameter t for correctness and completeness.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5b.extend([thinking5b, answer5b, feedback])
        thinking5b, answer5b = await cot_agent_5b(cot_inputs_5b, cot_reflect_instruction_5b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5b.id}, refining domain analysis, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {"thinking": thinking5b, "answer": answer5b}
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])

    debate_instruction_5c = (
        "Sub-task 5c: Perform a rigorous optimization of the function BD^2(t) over the valid domain of t, including detailed boundary behavior analysis "
        "to determine whether the minimum is attained or only approached asymptotically. Use Reflexion and Debate collaboration patterns to challenge and verify the optimization results and boundary analyses."
    )
    debate_agents_5c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5c = self.max_round
    all_thinking5c = [[] for _ in range(N_max_5c)]
    all_answer5c = [[] for _ in range(N_max_5c)]
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": debate_instruction_5c,
        "context": ["user query", "thinking of subtask 5b", "answer of subtask 5b"],
        "agent_collaboration": "Reflexion | Debate"
    }
    for r in range(N_max_5c):
        for i, agent in enumerate(debate_agents_5c):
            if r == 0:
                thinking5c, answer5c = await agent([taskInfo, thinking5b, answer5b], debate_instruction_5c, r, is_sub_task=True)
            else:
                input_infos_5c = [taskInfo, thinking5b, answer5b] + all_thinking5c[r-1] + all_answer5c[r-1]
                thinking5c, answer5c = await agent(input_infos_5c, debate_instruction_5c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, optimizing BD^2(t), thinking: {thinking5c.content}; answer: {answer5c.content}")
            all_thinking5c[r].append(thinking5c)
            all_answer5c[r].append(answer5c)
    final_decision_agent_5c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5c, answer5c = await final_decision_agent_5c([taskInfo] + all_thinking5c[-1] + all_answer5c[-1], "Sub-task 5c: Make final decision on BD^2(t) optimization.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding BD^2(t) optimization, thinking: {thinking5c.content}; answer: {answer5c.content}")
    sub_tasks.append(f"Sub-task 5c output: thinking - {thinking5c.content}; answer - {answer5c.content}")
    subtask_desc5c['response'] = {"thinking": thinking5c, "answer": answer5c}
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])

    cot_reflect_instruction_5d = (
        "Sub-task 5d: Validate the completeness and correctness of the optimization results from Subtask 5c. Cross-check intermediate expressions, domain considerations, and minimal values "
        "to ensure no oversight before proceeding to final answer selection."
    )
    cot_agent_5d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5d = self.max_round
    cot_inputs_5d = [taskInfo, thinking5c, answer5c]
    subtask_desc5d = {
        "subtask_id": "subtask_5d",
        "instruction": cot_reflect_instruction_5d,
        "context": ["user query", "thinking of subtask 5c", "answer of subtask 5c"],
        "agent_collaboration": "Reflexion"
    }
    thinking5d, answer5d = await cot_agent_5d(cot_inputs_5d, cot_reflect_instruction_5d, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5d.id}, validating optimization results, thinking: {thinking5d.content}; answer: {answer5d.content}")
    for i in range(N_max_5d):
        feedback, correct = await critic_agent_5d([taskInfo, thinking5d, answer5d],
                                                "Please review the validation of optimization results for correctness and completeness.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5d.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5d.extend([thinking5d, answer5d, feedback])
        thinking5d, answer5d = await cot_agent_5d(cot_inputs_5d, cot_reflect_instruction_5d, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5d.id}, refining validation, thinking: {thinking5d.content}; answer: {answer5d.content}")
    sub_tasks.append(f"Sub-task 5d output: thinking - {thinking5d.content}; answer - {answer5d.content}")
    subtask_desc5d['response'] = {"thinking": thinking5d, "answer": answer5d}
    logs.append(subtask_desc5d)
    print("Step 5d: ", sub_tasks[-1])

    cot_sc_instruction_6 = (
        "Sub-task 6: Select and verify the maximum lower bound for BD^2 found in Stage 3, ensuring it holds for all such rhombi inscribed on the hyperbola with diagonals intersecting at the origin. "
        "Provide the final answer with rigorous mathematical verification, including whether the minimum is attained or only an infimum, and discuss implications."
    )
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_6 = []
    thinkingmapping_6 = {}
    answermapping_6 = {}
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", "thinking of subtask 5d", "answer of subtask 5d"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking5d, answer5d], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, verifying maximum lower bound for BD^2, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6.content)
        thinkingmapping_6[answer6.content] = thinking6
        answermapping_6[answer6.content] = answer6
    answer6_content = Counter(possible_answers_6).most_common(1)[0][0]
    thinking6 = thinkingmapping_6[answer6_content]
    answer6 = answermapping_6[answer6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
