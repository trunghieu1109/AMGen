async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_N = self.max_sc
    debate_rounds = self.max_round

    # Stage 0: Formal definitions and conditions

    # Subtask 1: Define rectangle ABCD with symbolic coordinates and properties (SC_CoT)
    cot_sc_instruction_1 = (
        "Sub-task 1: Formally define rectangle ABCD with points A, B, C, D as symbolic coordinates. "
        "State rectangle properties (opposite sides equal, all angles 90 degrees). Record given side lengths AB=107 and BC=16. "
        "Do not fix orientation or coordinate placement."
    )
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking1, answer1 = await cot_sc_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, defining rectangle ABCD, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_answers_1 + possible_thinkings_1, "Sub-task 1: Synthesize and choose the most consistent definition of rectangle ABCD.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: Define rectangle EFGH with symbolic coordinates and properties (SC_CoT)
    cot_sc_instruction_2 = (
        "Sub-task 2: Formally define rectangle EFGH with points E, F, G, H as symbolic coordinates. "
        "State rectangle properties (opposite sides equal, all angles 90 degrees). Record given side lengths EF=184 and FG=17. "
        "Do not fix orientation; keep orientation as a parameter to explore later."
    )
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking2, answer2 = await cot_sc_agents_2[i]([taskInfo], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, defining rectangle EFGH, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent definition of rectangle EFGH.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    # Subtask 3: Express collinearity of D, E, C, F mathematically and analyze implications (SC_CoT)
    cot_sc_instruction_3 = (
        "Sub-task 3: Express the collinearity condition of points D, E, C, F mathematically. "
        "Derive the linear equation or parametric form these points satisfy. Analyze implications on relative positioning of rectangles ABCD and EFGH. "
        "Clarify constraints on coordinates of E and F relative to D and C. Avoid assuming order or direction without justification."
    )
    cot_sc_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking3, answer3 = await cot_sc_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3[i].id}, expressing collinearity, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_answers_3 + possible_thinkings_3, "Sub-task 3: Synthesize and choose the most consistent collinearity expression.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    # Subtask 4: Formulate cyclic quadrilateral condition using power-of-a-point or intersecting chords theorem (SC_CoT)
    cot_sc_instruction_4 = (
        "Sub-task 4: Formulate the cyclic quadrilateral condition for points A, D, H, G using the power-of-a-point theorem or intersecting chords theorem. "
        "Explicitly identify chords involved (e.g., ED·EC = EH·EG) and derive corresponding algebraic relation. "
        "Avoid using opposite angle sum property of rectangles as it is tautological."
    )
    cot_sc_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking4, answer4 = await cot_sc_agents_4[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_4[i].id}, formulating cyclic condition, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_answers_4 + possible_thinkings_4, "Sub-task 4: Synthesize and choose the most consistent cyclic quadrilateral condition.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    # Subtask 5: Identify assumptions and parameterize orientation of rectangle EFGH; explore both orientations (Debate)
    debate_instruction_5 = (
        "Sub-task 5: Identify and clearly state all necessary assumptions or coordinate placements to proceed with algebraic derivations. "
        "Consider both possible orientations of rectangle EFGH (e.g., FG vector pointing upward or downward). "
        "Parameterize orientation and prepare to explore alternatives systematically. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_5 = [[] for _ in range(debate_rounds)]
    all_answer_5 = [[] for _ in range(debate_rounds)]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(debate_rounds):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, exploring orientations, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking_5[r].append(thinking5)
            all_answer_5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking_5[-1] + all_answer_5[-1], "Sub-task 5: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc_5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    # Stage 1: Algebraic relations and integration

    # Subtask 6: Derive algebraic expressions relating coordinates of D, E, C, F on collinear line (SC_CoT)
    cot_sc_instruction_6 = (
        "Sub-task 6: Derive algebraic expressions relating coordinates of points D, E, C, F on the collinear line using known side lengths and rectangle properties. "
        "Express relations symbolically, maintaining parameters for orientation and position. Perform step-by-step symbolic expansions and simplifications with explicit checks to avoid arithmetic errors."
    )
    cot_sc_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_6 = []
    possible_thinkings_6 = []
    subtask_desc_6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", thinking3.content, answer3.content, thinking5.content, answer5.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking6, answer6 = await cot_sc_agents_6[i]([taskInfo, thinking3, answer3, thinking5, answer5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_6[i].id}, deriving algebraic relations, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6)
        possible_thinkings_6.append(thinking6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + possible_answers_6 + possible_thinkings_6, "Sub-task 6: Synthesize and choose the most consistent algebraic relations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc_6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc_6)
    print("Step 6: ", sub_tasks[-1])

    # Subtask 7: Translate cyclic quadrilateral condition into explicit algebraic equations involving coordinates or lengths (Reflexion)
    reflect_inst_7 = (
        "Sub-task 7: Translate the cyclic quadrilateral condition from Sub-task 4 into explicit algebraic equations involving coordinates or lengths derived in Sub-task 6, integrating known side lengths EF and FG. "
        "Perform careful symbolic manipulation and verify each step numerically to prevent arithmetic slips. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking4, answer4, thinking6, answer6]
    subtask_desc_7 = {
        "subtask_id": "subtask_7",
        "instruction": reflect_inst_7,
        "context": ["user query", thinking4.content, answer4.content, thinking6.content, answer6.content],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, reflect_inst_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, translating cyclic condition, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        feedback7, correct7 = await critic_agent_7([taskInfo, thinking7, answer7], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback7])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, reflect_inst_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining solution, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc_7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc_7)
    print("Step 7: ", sub_tasks[-1])

    # Subtask 8: Combine algebraic relations from collinearity and cyclic conditions into consistent system (Reflexion)
    reflect_inst_8 = (
        "Sub-task 8: Combine algebraic relations from collinearity and cyclic conditions to form a consistent system linking unknown length CE with known side lengths and geometric constraints. "
        "Prepare system for solution, ensuring all parameters and assumptions are explicitly included. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_8 = self.max_round
    cot_inputs_8 = [taskInfo, thinking6, answer6, thinking7, answer7]
    subtask_desc_8 = {
        "subtask_id": "subtask_8",
        "instruction": reflect_inst_8,
        "context": ["user query", thinking6.content, answer6.content, thinking7.content, answer7.content],
        "agent_collaboration": "Reflexion"
    }
    thinking8, answer8 = await cot_agent_8(cot_inputs_8, reflect_inst_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, combining relations, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(N_max_8):
        feedback8, correct8 = await critic_agent_8([taskInfo, thinking8, answer8], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_8.id}, providing feedback, thinking: {feedback8.content}; answer: {correct8.content}")
        if correct8.content == "True":
            break
        cot_inputs_8.extend([thinking8, answer8, feedback8])
        thinking8, answer8 = await cot_agent_8(cot_inputs_8, reflect_inst_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refining combined system, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc_8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc_8)
    print("Step 8: ", sub_tasks[-1])

    # Subtask 9: Geometric validation of solution candidates (Reflexion)
    reflect_inst_9 = (
        "Sub-task 9: Perform comprehensive geometric validation of current solution candidates. "
        "Verify E, F, G, H form rectangle with given side lengths and right angles; confirm A, D, H, G lie on same circle by checking equal distances from computed center; confirm collinearity of D, E, C, F. "
        "If inconsistencies found, identify assumptions or orientations to reconsider. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_9 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_9 = self.max_round
    cot_inputs_9 = [taskInfo, thinking8, answer8]
    subtask_desc_9 = {
        "subtask_id": "subtask_9",
        "instruction": reflect_inst_9,
        "context": ["user query", thinking8.content, answer8.content],
        "agent_collaboration": "Reflexion"
    }
    thinking9, answer9 = await cot_agent_9(cot_inputs_9, reflect_inst_9, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_9.id}, validating geometry, thinking: {thinking9.content}; answer: {answer9.content}")
    for i in range(N_max_9):
        feedback9, correct9 = await critic_agent_9([taskInfo, thinking9, answer9], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_9.id}, providing feedback, thinking: {feedback9.content}; answer: {correct9.content}")
        if correct9.content == "True":
            break
        cot_inputs_9.extend([thinking9, answer9, feedback9])
        thinking9, answer9 = await cot_agent_9(cot_inputs_9, reflect_inst_9, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_9.id}, refining validation, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc_9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc_9)
    print("Step 9: ", sub_tasks[-1])

    # Stage 2: Final solving

    # Subtask 10: Solve system explicitly to compute length CE and verify all constraints (CoT)
    cot_instruction_10 = (
        "Sub-task 10: Based on validated system and geometric consistency checks, solve system explicitly to compute length CE. "
        "Verify computed CE satisfies all geometric constraints and solution is consistent with all assumptions and validations. "
        "If multiple orientation cases exist, select one consistent with all constraints."
    )
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_10 = {
        "subtask_id": "subtask_10",
        "instruction": cot_instruction_10,
        "context": ["user query", thinking9.content, answer9.content],
        "agent_collaboration": "CoT"
    }
    thinking10, answer10 = await cot_agent_10([taskInfo, thinking9, answer9], cot_instruction_10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_10.id}, solving final system, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc_10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc_10)
    print("Step 10: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs
