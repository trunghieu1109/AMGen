async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Derive explicit algebraic or parametric representations for points A, B, C, and D on the hyperbola x^2/20 - y^2/24 = 1 "
        "such that the diagonals of the rhombus intersect at the origin. Use the midpoint condition to express B = -D and A = -C. "
        "Validate these representations satisfy the hyperbola equation and midpoint condition. Avoid assuming any specific parameterization without justification. "
        "Emphasize clarity in notation and ensure the symmetry and hyperbola constraints are explicitly stated and verified."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, deriving parametric representations, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_1 = (
        "Sub-task 2: Formulate the rhombus conditions algebraically: (a) equal side lengths, and (b) perpendicular diagonals. "
        "Express the side length in terms of the coordinates of A, B, C, and D, and write the perpendicularity condition of the diagonals as a dot product equal to zero. "
        "Ensure these conditions are consistent with the point representations from subtask_1. Avoid mixing up diagonal endpoints or misapplying vector properties. "
        "Explicitly state all constraints to be used in subsequent subtasks."
    )
    N_sc_1 = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1)]
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    subtask_desc_1 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1):
        thinking_1, answer_1 = await cot_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, formulating rhombus conditions, thinking: {thinking_1.content}; answer: {answer_1.content}")
        possible_answers_1.append(answer_1.content)
        thinkingmapping_1[answer_1.content] = thinking_1
        answermapping_1[answer_1.content] = answer_1
    best_answer_1 = Counter(possible_answers_1).most_common(1)[0][0]
    thinking_1 = thinkingmapping_1[best_answer_1]
    answer_1 = answermapping_1[best_answer_1]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_debate_instruction_2a = (
        "Sub-task 3a: Derive an explicit equation from the equal side length condition to express one parameter (e.g., s) in terms of the other (t), or vice versa. "
        "Incorporate all constraints from the hyperbola equation, perpendicularity of diagonals, and equal side lengths. Carefully handle nonlinearities and ensure no constraints are omitted. "
        "This subtask must produce a clear, explicit relation between parameters that fully encodes the rhombus conditions. "
        "Use debate-style collaboration to challenge and verify the derivation."
    )
    debate_agents_2a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2a = self.max_round
    all_thinking_2a = [[] for _ in range(N_max_2a)]
    all_answer_2a = [[] for _ in range(N_max_2a)]
    subtask_desc_2a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_debate_instruction_2a,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2a):
        for i, agent in enumerate(debate_agents_2a):
            if r == 0:
                thinking_2a, answer_2a = await agent([taskInfo, thinking_1, answer_1], cot_sc_debate_instruction_2a, r, is_sub_task=True)
            else:
                input_infos_2a = [taskInfo, thinking_1, answer_1] + all_thinking_2a[r-1] + all_answer_2a[r-1]
                thinking_2a, answer_2a = await agent(input_infos_2a, cot_sc_debate_instruction_2a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, deriving parameter relation, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
            all_thinking_2a[r].append(thinking_2a)
            all_answer_2a[r].append(answer_2a)
    thinking_2a, answer_2a = all_thinking_2a[-1][0], all_answer_2a[-1][0]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 3a: ", sub_tasks[-1])

    cot_sc_instruction_2b = (
        "Sub-task 3b: Substitute the parameter relation derived in subtask_3a into the expression for BD^2 to obtain a single-variable function f(t) (or f(s)). "
        "Simplify the expression as much as possible, ensuring all constraints are incorporated. Prepare this function for rigorous optimization. "
        "Avoid losing any constraints during substitution. Explicitly carry forward_14 all key relations and verify symbolic consistency."
    )
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc_2b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", thinking_2a.content, answer_2a.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_2b, answer_2b = await cot_agents_2b[i]([taskInfo, thinking_2a, answer_2a], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, substituting parameter relation into BD^2, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
        possible_answers_2b.append(answer_2b.content)
        thinkingmapping_2b[answer_2b.content] = thinking_2b
        answermapping_2b[answer_2b.content] = answer_2b
    best_answer_2b = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking_2b = thinkingmapping_2b[best_answer_2b]
    answer_2b = answermapping_2b[best_answer_2b]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)
    print("Step 3b: ", sub_tasks[-1])

    debate_reflexion_instruction_2c = (
        "Sub-task 3c: Perform automatic consistency checks on the parameter relations and the function f(t) to verify that candidate parameter values satisfy all constraints (hyperbola, equal sides, perpendicular diagonals). "
        "Identify and discard any spurious or extraneous solutions before proceeding to optimization. This step prevents propagation of errors and ensures reliability of subsequent analysis. "
        "Use debate and reflexion collaboration to rigorously verify all constraints."
    )
    debate_agents_2c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2c = self.max_round
    all_thinking_2c = [[] for _ in range(N_max_2c)]
    all_answer_2c = [[] for _ in range(N_max_2c)]
    subtask_desc_2c = {
        "subtask_id": "subtask_3c",
        "instruction": debate_reflexion_instruction_2c,
        "context": ["user query", thinking_2b.content, answer_2b.content],
        "agent_collaboration": "Debate | Reflexion"
    }
    for r in range(N_max_2c):
        for i, agent in enumerate(debate_agents_2c):
            if r == 0:
                thinking_2c, answer_2c = await agent([taskInfo, thinking_2b, answer_2b], debate_reflexion_instruction_2c, r, is_sub_task=True)
            else:
                input_infos_2c = [taskInfo, thinking_2b, answer_2b] + all_thinking_2c[r-1] + all_answer_2c[r-1]
                thinking_2c, answer_2c = await agent(input_infos_2c, debate_reflexion_instruction_2c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, consistency checking, thinking: {thinking_2c.content}; answer: {answer_2c.content}")
            all_thinking_2c[r].append(thinking_2c)
            all_answer_2c[r].append(answer_2c)
    thinking_2c, answer_2c = all_thinking_2c[-1][0], all_answer_2c[-1][0]
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking_2c.content}; answer - {answer_2c.content}")
    subtask_desc_2c['response'] = {"thinking": thinking_2c, "answer": answer_2c}
    logs.append(subtask_desc_2c)
    print("Step 3c: ", sub_tasks[-1])

    cot_sc_debate_instruction_3 = (
        "Sub-task 4: Analyze the function f(t) = BD^2 rigorously to find its supremum under the given constraints. "
        "This includes: (a) computing the derivative d/dt f(t), (b) solving for critical points, (c) analyzing second derivatives or using other tests to confirm maxima, and (d) examining boundary cases to determine if the supremum is attained or only approached. "
        "Confirm that candidate maxima correspond to valid rhombi on the hyperbola with diagonals intersecting at the origin. Avoid assuming the supremum is attained without verification. "
        "Use iterative or debate-style collaboration to challenge and verify each step."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc_3 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_debate_instruction_3,
        "context": ["user query", thinking_2c.content, answer_2c.content],
        "agent_collaboration": "SC_CoT | Debate | Reflexion"
    }
    for i in range(self.max_sc):
        thinking_3, answer_3 = await cot_agents_3[i]([taskInfo, thinking_2c, answer_2c], cot_sc_debate_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, analyzing supremum of BD^2, thinking: {thinking_3.content}; answer: {answer_3.content}")
        possible_answers_3.append(answer_3.content)
        thinkingmapping_3[answer_3.content] = thinking_3
        answermapping_3[answer_3.content] = answer_3
    best_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking_3 = thinkingmapping_3[best_answer_3]
    answer_3 = answermapping_3[best_answer_3]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 4: ", sub_tasks[-1])

    reflect_instr_4 = (
        "Sub-task 5: Select the greatest real number less than or equal to the supremum of BD^2 found in subtask_4, ensuring it is valid for all such rhombi. "
        "Verify the solution by: (a) checking boundary cases, (b) confirming no larger value of BD^2 satisfies all constraints, (c) providing explicit proof or counterexamples regarding whether the supremum is attained or only a limit, and (d) performing numeric approximations or alternative methods to cross-check symbolic results. "
        "Provide the final answer along with a detailed verification summary. Avoid ambiguity in interpreting 'greatest real number less than BD^2 for all such rhombi'. Encourage reflexion and debate to rigorously validate the final result."
    )
    cot_reflect_instruction_4 = "Sub-task 5: Your problem is to finalize and verify the supremum of BD^2." + reflect_instr_4
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking_3, answer_3]
    subtask_desc_4 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", thinking_3.content, answer_3.content],
        "agent_collaboration": "Reflexion | Debate | CoT"
    }
    thinking_4, answer_4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, verifying supremum and final answer, thinking: {thinking_4.content}; answer: {answer_4.content}")
    for i in range(N_max_4):
        feedback_4, correct_4 = await critic_agent_4([taskInfo, thinking_4, answer_4], "Please review and provide limitations of the provided solution. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback_4.content}; answer: {correct_4.content}")
        if correct_4.content == "True":
            break
        cot_inputs_4.extend([thinking_4, answer_4, feedback_4])
        thinking_4, answer_4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining final answer, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_4, answer_4, sub_tasks, agents)
    return final_answer, logs
