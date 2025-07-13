async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Identify and verify the geometric properties of the convex equilateral hexagon ABCDEF, "
        "specifically that all sides are equal in length and that opposite sides are parallel. "
        "Formulate the vector identity AB + CD + EF = 0 without assuming fixed angles, and set up coordinate parametrizations "
        "for points A, C, and E and the directions of sides AB, CD, and EF using free parameters."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying hexagon properties, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Analyze the triangle formed by the extensions of sides AB, CD, and EF. "
        "Express the lines AB, CD, and EF parametrically in terms of the hexagon side length s and direction parameters from Subtask 1, "
        "preparing for intersection computations."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing triangle lines parametrization, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Compute the intersection points of the pairs of extended lines (AB & CD, CD & EF, EF & AB) symbolically "
        "in terms of the hexagon side length s and the free direction parameters. Derive algebraic expressions for the vertices of the triangle formed by these intersections."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, computing intersection points, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Formulate a system of three equations expressing the distances between the intersection points obtained in Subtask 3 "
        "as 200, 240, and 300 respectively. Use symbolic or numeric solvers to handle the algebraic complexity and solve for the hexagon side length s "
        "and the direction parameters simultaneously."
    )
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
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, solving system of equations, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: Conduct a debate among multiple agents proposing and critiquing different parametrizations and geometric configurations "
        "of the directions of sides AB, CD, and EF. Evaluate the validity of assumptions, explore alternative angle relationships, "
        "and identify contradictions or inconsistencies in the solutions from Subtask 4."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating parametrizations, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on parametrizations and contradictions.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding parametrizations, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_reflect_instruction_6 = (
        "Sub-task 6: Integrate reflexion and critic feedback to identify and resolve contradictions found in the side length comparisons "
        "and direction parametrizations. Enforce iterative refinement of assumptions and recalculations as needed, ensuring consistency "
        "with all given data and geometric constraints."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, resolving contradictions, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6],
                                               "please review the contradiction resolution and refinements.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining contradiction resolution, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_7 = (
        "Sub-task 7: Implement conditional branching or iterative loops triggered by contradictions or inconsistencies detected in Subtask 6. "
        "Re-execute relevant subtasks with revised assumptions or alternative parametrizations until a consistent solution is found."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, managing iterative refinement, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    cot_reflect_instruction_8 = (
        "Sub-task 8: Verify numerically that the computed hexagon side length s and direction parameters produce a triangle with side lengths exactly 200, 240, and 300. "
        "Confirm that all geometric constraints of the hexagon and triangle are satisfied, and provide the final validated answer for the hexagon side length."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_8 = self.max_round
    cot_inputs_8 = [taskInfo, thinking7, answer7]
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_reflect_instruction_8,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "Reflexion"
    }
    thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, verifying final solution, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(N_max_8):
        feedback, correct = await critic_agent_8([taskInfo, thinking8, answer8],
                                               "please review the final verification and confirm correctness.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_8.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_8.extend([thinking8, answer8, feedback])
        thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refining final verification, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs
