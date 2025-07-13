async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    cot_sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]

    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)

    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]

    # Sub-task 1: Extract and clearly list all given numerical values and geometric properties
    cot_instruction_1 = (
        "Sub-task 1: Extract and clearly list all given numerical values and geometric properties from the problem statement, "
        "including side lengths of rectangles ABCD and EFGH, the collinearity condition of points D, E, C, F, and the concyclicity of points A, D, H, G. "
        "Avoid making any assumptions about point order, rectangle orientation, or coordinate placement at this stage. "
        "Present the data in a structured format suitable for further processing."
    )
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting given data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    # Sub-task 2: Transform given side lengths and geometric properties into explicit geometric facts
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, transform the given side lengths and geometric properties into explicit geometric facts, "
        "such as listing all side lengths of rectangles ABCD and EFGH, stating right angle properties, and summarizing implications of rectangles having equal opposite sides. "
        "Do not assign coordinates or assume point order. Prepare these facts as formal constraints for later use."
    )
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2 = []
    possible_thinkings_2 = []
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_sc_agents[i]([taskInfo, thinking1.content, answer1.content], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, transforming data into geometric facts, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent geometric facts.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)

    # Sub-task 3: Enumerate all possible linear orders of points D, E, C, F
    cot_instruction_3 = (
        "Sub-task 3: Enumerate all possible linear orders of points D, E, C, and F on their common line. "
        "For each order, analyze how segment lengths and relative positions affect the rectangles’ orientations and the segment CE. "
        "Avoid fixing any single order prematurely. Present a complete list of candidate orders with symbolic or parametric expressions for segment relationships."
    )
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent([taskInfo, thinking2.content, answer2.content], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, enumerating point orders, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    # Sub-task 4: Identify and enumerate all geometric constraints implied by concyclicity of A, D, H, G
    cot_instruction_4 = (
        "Sub-task 4: Identify and enumerate all geometric constraints implied by the concyclicity of points A, D, H, and G. "
        "Include properties of cyclic quadrilaterals such as opposite angles summing to 180°, power of a point, and chord properties. "
        "Avoid coordinate assignments or angle measures. Prepare these constraints symbolically for integration with collinearity conditions."
    )
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent([taskInfo, thinking2.content, answer2.content], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, enumerating concyclicity constraints, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)

    # Sub-task 5: Combine constraints from collinearity and concyclicity to validate or eliminate candidate orders
    cot_sc_instruction_5 = (
        "Sub-task 5: Combine the constraints from collinearity (Sub-task 3) and concyclicity (Sub-task 4) to validate or eliminate candidate orders of points D, E, C, and F. "
        "Use logical and geometric reasoning to prune impossible or inconsistent configurations. Present the reduced set of feasible orders with justification. "
        "Avoid solving equations yet; focus on logical consistency."
    )
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", thinking3.content, answer3.content, thinking4.content, answer4.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_5 = []
    possible_thinkings_5 = []
    for i in range(self.max_sc):
        thinking5, answer5 = await cot_sc_agents[i]([taskInfo, thinking3.content, answer3.content, thinking4.content, answer4.content], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, validating candidate orders, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5)
        possible_thinkings_5.append(thinking5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + possible_answers_5 + possible_thinkings_5, "Sub-task 5: Synthesize and choose the most consistent feasible point orders.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)

    # Sub-task 6: Assign coordinate systems consistent with validated constraints and feasible point orders
    cot_instruction_6 = (
        "Sub-task 6: Assign coordinate systems or geometric models for points A, B, C, D, E, F, G, H consistent with all previously validated constraints and feasible point orders from Sub-task 5. "
        "If multiple orders remain, implement branching coordinate assignments or parameterize the order to maintain all viable cases. "
        "Ensure rectangle properties, collinearity of D, E, C, F, and concyclicity of A, D, H, G are satisfied. Avoid arbitrary placements violating constraints."
    )
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking5.content, answer5.content],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent([taskInfo, thinking5.content, answer5.content], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, assigning coordinates, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)

    # Sub-task 7: Formulate the circle equation step-by-step using assigned coordinates
    cot_instruction_7 = (
        "Sub-task 7: Formulate the circle equation step-by-step using the coordinates assigned in Sub-task 6. "
        "Substitute points H and G into the circle equation to derive a quadratic equation in the coordinate(s) of point E. "
        "Provide a detailed algebraic derivation of the coefficients and the quadratic form, ensuring transparency and correctness. "
        "Avoid skipping steps or asserting results without justification."
    )
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", thinking6.content, answer6.content],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent([taskInfo, thinking6.content, answer6.content], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, deriving circle equation, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)

    # Sub-task 8: Solve the quadratic equation carefully and compute length CE
    cot_instruction_8 = (
        "Sub-task 8: Solve the quadratic equation derived in Sub-task 7 carefully, selecting the physically meaningful root(s) consistent with geometric constraints and the validated point order(s). "
        "Compute the length CE accordingly. Present all algebraic steps and verify the solution’s consistency with rectangle properties, collinearity, and concyclicity. "
        "Avoid premature selection of roots without justification."
    )
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", thinking7.content, answer7.content],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent([taskInfo, thinking7.content, answer7.content], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, solving quadratic and computing CE, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)

    # Sub-task 9: Critically verify all algebraic and geometric steps
    reflect_inst_9 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_9 = f"Sub-task 9: Critically verify all algebraic and geometric steps from Sub-tasks 6 through 8, including correctness of coordinate assignments, circle equation derivation, quadratic solution, and computed length CE. Check for internal consistency and adherence to all constraints. If contradictions or inconsistencies are found, trigger a feedback loop to revisit earlier subtasks, especially point order assumptions and coordinate assignments. {reflect_inst_9}"
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_reflect_instruction_9,
        "context": ["user query", thinking6.content, answer6.content, thinking7.content, answer7.content, thinking8.content, answer8.content],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_9 = [taskInfo, thinking6.content, answer6.content, thinking7.content, answer7.content, thinking8.content, answer8.content]
    thinking9, answer9 = await cot_agent(cot_inputs_9, cot_reflect_instruction_9, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, verifying algebraic and geometric steps, thinking: {thinking9.content}; answer: {answer9.content}")
    critic_inst_9 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback9, correct9 = await critic_agent([taskInfo, thinking9.content, answer9.content], critic_inst_9, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback9.content}; answer: {correct9.content}")
        if correct9.content == "True":
            break
        cot_inputs_9.extend([thinking9.content, answer9.content, feedback9.content])
        thinking9, answer9 = await cot_agent(cot_inputs_9, cot_reflect_instruction_9, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining verification, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)

    # Sub-task 10: Conduct a collaborative debate to assess assumptions, derivations, and final results
    debate_instr_10 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_10 = f"Sub-task 10: Conduct a collaborative debate to assess assumptions, derivations, and final results. {debate_instr_10}"
    N_max_10 = self.max_round
    all_thinking_10 = [[] for _ in range(N_max_10)]
    all_answer_10 = [[] for _ in range(N_max_10)]
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": debate_instruction_10,
        "context": ["user query", thinking9.content, answer9.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_10):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking10, answer10 = await agent([taskInfo, thinking9.content, answer9.content], debate_instruction_10, r, is_sub_task=True)
            else:
                input_infos_10 = [taskInfo, thinking9.content, answer9.content] + all_thinking_10[r-1] + all_answer_10[r-1]
                thinking10, answer10 = await agent(input_infos_10, debate_instruction_10, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating final solution, thinking: {thinking10.content}; answer: {answer10.content}")
            all_thinking_10[r].append(thinking10)
            all_answer_10[r].append(answer10)
    final_decision_agent_10 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking10, answer10 = await final_decision_agent_10([taskInfo] + all_thinking_10[-1] + all_answer_10[-1], "Sub-task 10: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final CE, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)

    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs
