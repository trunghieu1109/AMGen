async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = (
        "Sub-task 1: Analyze the geometric properties of the convex equilateral hexagon ABCDEF with parallel opposite sides, "
        "establish that the hexagon is centrally symmetric, and identify implications for the directions and relationships of its side vectors without assuming cyclicity or regularity."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing geometric properties, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = (
        "Sub-task 2: Assign direction angles to three independent side vectors of the hexagon (e.g., vectors for sides AB, BC, and CD), "
        "express all six side vectors in terms of these angles and the common side length s, and write the vector closure condition ensuring the hexagon is closed, "
        "based on the analysis from Sub-task 1."
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, assigning direction angles and expressing side vectors, thinking: {thinking2.content}; answer: {answer2.content}")
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
    
    cot_sc_instruction_3 = (
        "Sub-task 3: Derive explicit expressions for the intersection points of the extended lines AB, CD, and EF in terms of the hexagon side length s and the assigned direction angles, "
        "carefully modeling the triangle formed by these intersections as the triangle with sides 200, 240, and 300. "
        "Emphasize that the triangle sides are distances between intersection points, not single hexagon sides, and use vector sums accordingly."
    )
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
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, deriving intersection points and triangle modeling, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_reflect_instruction_3 = (
        "Sub-task 3 Reflexion: Critically evaluate the dimensional consistency, logical validity, and mathematical correctness of the derived expressions for the intersection points and triangle modeling from Sub-task 3. "
        "Identify any inconsistencies or errors and refine the reasoning accordingly."
    )
    cot_agent_reflect_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round
    cot_inputs_3 = [taskInfo, thinking2, answer2, thinking3, answer3]
    subtask_desc3_reflect = {
        "subtask_id": "subtask_3_reflexion",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Reflexion"
    }
    thinking3_reflect, answer3_reflect = await cot_agent_reflect_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflect_3.id}, evaluating intersection expressions, thinking: {thinking3_reflect.content}; answer: {answer3_reflect.content}")
    for i in range(N_max_reflect):
        feedback, correct = await critic_agent_3([taskInfo, thinking3_reflect, answer3_reflect],
                                               "Please review the dimensional consistency and correctness of the intersection point expressions and provide limitations.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3_reflect, answer3_reflect, feedback])
        thinking3_reflect, answer3_reflect = await cot_agent_reflect_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflect_3.id}, refining intersection expressions, thinking: {thinking3_reflect.content}; answer: {answer3_reflect.content}")
    sub_tasks.append(f"Sub-task 3 Reflexion output: thinking - {thinking3_reflect.content}; answer - {answer3_reflect.content}")
    subtask_desc3_reflect['response'] = {
        "thinking": thinking3_reflect,
        "answer": answer3_reflect
    }
    logs.append(subtask_desc3_reflect)
    print("Step 3 Reflexion: ", sub_tasks[-1])
    
    cot_sc_instruction_4 = (
        "Sub-task 4: Formulate equations for the lengths of the triangle sides (200, 240, 300) as functions of s and the direction angles by computing distances between the intersection points derived in Sub-task 3, "
        "ensuring dimensional consistency and correct geometric interpretation."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 3 reflexion", "answer of subtask 3 reflexion"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3_reflect, answer3_reflect], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, formulating equations for triangle side lengths, thinking: {thinking4.content}; answer: {answer4.content}")
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
    
    cot_reflect_instruction_4 = (
        "Sub-task 4 Reflexion: Critically evaluate the system of equations relating the triangle side lengths to s and direction angles, "
        "checking for dimensional consistency, correctness, and feasibility. Refine the equations or assumptions if necessary."
    )
    cot_agent_reflect_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_4 = [taskInfo, thinking4, answer4]
    subtask_desc4_reflect = {
        "subtask_id": "subtask_4_reflexion",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Reflexion"
    }
    thinking4_reflect, answer4_reflect = await cot_agent_reflect_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflect_4.id}, evaluating system of equations, thinking: {thinking4_reflect.content}; answer: {answer4_reflect.content}")
    for i in range(N_max_reflect):
        feedback, correct = await critic_agent_4([taskInfo, thinking4_reflect, answer4_reflect],
                                               "Please review the system of equations for correctness and provide limitations.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4_reflect, answer4_reflect, feedback])
        thinking4_reflect, answer4_reflect = await cot_agent_reflect_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflect_4.id}, refining system of equations, thinking: {thinking4_reflect.content}; answer: {answer4_reflect.content}")
    sub_tasks.append(f"Sub-task 4 Reflexion output: thinking - {thinking4_reflect.content}; answer - {answer4_reflect.content}")
    subtask_desc4_reflect['response'] = {
        "thinking": thinking4_reflect,
        "answer": answer4_reflect
    }
    logs.append(subtask_desc4_reflect)
    print("Step 4 Reflexion: ", sub_tasks[-1])
    
    debate_instruction_5 = (
        "Sub-task 5: Solve the system of equations from Sub-task 4 to find the numeric value of the hexagon's side length s, "
        "verifying the solution by checking that the computed triangle side lengths match the given values and that all geometric constraints are satisfied. "
        "Engage in a debate to test candidate solutions for accuracy and consistency."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4 reflexion", "answer of subtask 4 reflexion"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4_reflect, answer4_reflect], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4_reflect, answer4_reflect] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, solving for hexagon side length, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the numeric value of the hexagon's side length s.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final hexagon side length, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
