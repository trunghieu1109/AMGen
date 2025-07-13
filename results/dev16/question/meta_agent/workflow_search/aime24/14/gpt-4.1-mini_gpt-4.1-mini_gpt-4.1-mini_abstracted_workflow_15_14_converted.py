async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Parametrization and geometric properties

    # Subtask 1: Parametrize points A, B, C, D on the hyperbola with diagonals intersecting at origin
    cot_instruction_1 = (
        "Sub-task 1: Formally represent points A, B, C, and D on the hyperbola x^2/20 - y^2/24 = 1, "
        "explicitly stating the parametrization of points on the hyperbola. Include the condition that the diagonals of rhombus ABCD intersect at the origin, emphasizing that the origin is the midpoint of both diagonals and that points B and D, as well as A and C, are symmetric with respect to the origin."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, parametrizing points and midpoint condition, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: Formalize rhombus properties relevant to the problem
    cot_instruction_2 = (
        "Sub-task 2: State and formalize the geometric properties of rhombus ABCD relevant to the problem: equal side lengths, diagonals bisecting each other at right angles, "
        "and the implications of these properties on the coordinates of points A, B, C, and D. Avoid assuming any orientation or labeling beyond what is necessary for symmetry and midpoint conditions."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, formalizing rhombus properties, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Subtask 3: Combine hyperbola constraint and rhombus properties to derive algebraic relations
    cot_sc_instruction_3 = (
        "Sub-task 3: Combine the hyperbola constraint and rhombus properties to derive algebraic relations between the parameters describing points A, B, C, and D. "
        "Exploit symmetry and midpoint conditions to simplify the problem, ensuring that the parametrization respects the hyperbola equation and the midpoint at the origin."
    )
    N_sc = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking1.content, thinking2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, thinking2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, deriving algebraic relations, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_thinkings_3, "Sub-task 3: Synthesize and choose the most consistent algebraic relations for the problem", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing algebraic relations, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Derive perpendicularity condition and express BD^2

    # Subtask 1: Derive perpendicularity condition of diagonals with hyperbolic parametrization
    cot_instruction_4 = (
        "Sub-task 1: Derive the perpendicularity condition of the diagonals from the dot product constraint, correctly substituting the hyperbolic parametrization x = sqrt(20) cosh t, y = sqrt(24) sinh t. "
        "Carefully re-derive the key identity involving tanh u and tanh v, preserving sign information and ensuring that the parameters correspond to points on the hyperbola with opposite signs as required by the problem's symmetry."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, deriving perpendicularity condition, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Subtask 2: Express BD^2 in terms of hyperbolic parameters with corrected constraint
    cot_instruction_5 = (
        "Sub-task 2: Express the squared length of diagonal BD in terms of the hyperbolic parameters derived earlier, incorporating the corrected perpendicularity constraint. "
        "Formulate the expression for BD^2 explicitly and prepare it for analysis under the constraint."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking3.content, thinking4.content],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, thinking4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, expressing BD^2, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Subtask 3: Formulate equal side length condition in terms of hyperbolic parameters
    cot_sc_instruction_6 = (
        "Sub-task 3: Formulate the equal side length condition of the rhombus in terms of the hyperbolic parameters and relate it to the diagonal lengths. "
        "Derive an equation linking these quantities that respects all geometric and hyperbolic constraints, ensuring no algebraic errors in the process."
    )
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_6 = []
    possible_thinkings_6 = []
    subtask_desc6 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", thinking3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking3], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, formulating equal side length condition, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6)
        possible_thinkings_6.append(thinking6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + possible_thinkings_6, "Sub-task 3: Synthesize and choose the most consistent equal side length condition", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing equal side length condition, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    # Subtask 4: Combine expressions to relate BD^2 to parameters incorporating perpendicularity
    cot_sc_instruction_7 = (
        "Sub-task 4: Combine the expressions from subtasks 2 and 3 of this stage to obtain a single equation or system that relates BD^2 to the parameters defining the rhombus on the hyperbola, incorporating the corrected perpendicularity condition. "
        "Ensure the system is consistent and ready for optimization analysis."
    )
    cot_agents_7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_7 = []
    possible_thinkings_7 = []
    subtask_desc7 = {
        "subtask_id": "stage_2.subtask_4",
        "instruction": cot_sc_instruction_7,
        "context": ["user query", thinking5.content, thinking6.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking7, answer7 = await cot_agents_7[i]([taskInfo, thinking5, thinking6], cot_sc_instruction_7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_7[i].id}, combining expressions for BD^2, thinking: {thinking7.content}; answer: {answer7.content}")
        possible_answers_7.append(answer7)
        possible_thinkings_7.append(thinking7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + possible_thinkings_7, "Sub-task 4: Synthesize and finalize system relating BD^2 to parameters", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing system for BD^2, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    # Stage 3: Analyze and verify infimum of BD^2

    # Subtask 1: Analyze the system to identify the infimum of BD^2
    debate_instruction_8 = (
        "Sub-task 1: Analyze the derived system to identify the infimum (greatest lower bound) of BD^2 for all rhombi inscribed in the hyperbola with diagonals intersecting at the origin. "
        "Focus on minimizing BD^2 under the corrected constraints, and carefully examine boundary behavior as parameters approach critical values, avoiding misinterpretation of supremum versus infimum. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_8 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_8 = self.max_round
    all_thinking8 = [[] for _ in range(N_max_8)]
    all_answer8 = [[] for _ in range(N_max_8)]
    subtask_desc8 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_8,
        "context": ["user query", thinking7.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_8):
        for i, agent in enumerate(debate_agents_8):
            if r == 0:
                thinking8, answer8 = await agent([taskInfo, thinking7], debate_instruction_8, r, is_sub_task=True)
            else:
                input_infos_8 = [taskInfo, thinking7] + all_thinking8[r-1]
                thinking8, answer8 = await agent(input_infos_8, debate_instruction_8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing infimum of BD^2, thinking: {thinking8.content}; answer: {answer8.content}")
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)
    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + all_thinking8[-1], "Sub-task 1: Finalize analysis of infimum of BD^2. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing infimum analysis, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    # Subtask 2: Verify the infimum by checking boundary cases
    reflect_inst_9 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_9 = (
        "Sub-task 2: Verify the infimum found for BD^2 by checking boundary cases and confirming that no smaller value is possible under the problem's constraints. "
        "Explicitly confirm that the infimum is approached from above and that the value is consistent with the hyperbola and rhombus properties. " + reflect_inst_9
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_9 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_9 = self.max_round
    cot_inputs_9 = [taskInfo, thinking8]
    subtask_desc9 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": cot_reflect_instruction_9,
        "context": ["user query", thinking8.content],
        "agent_collaboration": "Reflexion"
    }
    thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_9.id}, verifying infimum, thinking: {thinking9.content}; answer: {answer9.content}")
    for i in range(N_max_9):
        feedback9, correct9 = await critic_agent_9([taskInfo, thinking9], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_9.id}, providing feedback, thinking: {feedback9.content}; answer: {correct9.content}")
        if correct9.content == "True":
            break
        cot_inputs_9.extend([thinking9, feedback9])
        thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_9.id}, refining verification, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    # Subtask 3: Clearly state the greatest real number less than BD^2 (infimum) with summary
    cot_sc_instruction_10 = (
        "Sub-task 3: Clearly state the greatest real number less than BD^2 for all such rhombi, i.e., the infimum of BD^2, without rounding or truncation. "
        "Distinguish explicitly between supremum and infimum in the final answer, and provide a concise summary of the reasoning leading to this conclusion."
    )
    cot_agents_10 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_10 = []
    possible_thinkings_10 = []
    subtask_desc10 = {
        "subtask_id": "stage_3.subtask_3",
        "instruction": cot_sc_instruction_10,
        "context": ["user query", thinking9.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking10, answer10 = await cot_agents_10[i]([taskInfo, thinking9], cot_sc_instruction_10, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_10[i].id}, stating final infimum, thinking: {thinking10.content}; answer: {answer10.content}")
        possible_answers_10.append(answer10)
        possible_thinkings_10.append(thinking10)
    final_decision_agent_10 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking10, answer10 = await final_decision_agent_10([taskInfo] + possible_thinkings_10, "Sub-task 3: Finalize and clearly state the infimum of BD^2 with reasoning summary", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing answer, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs
