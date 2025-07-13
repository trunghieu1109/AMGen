async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = "Sub-task 1: Identify and explicitly list all given elements and constraints: the two rectangles ABCD and EFGH, their side lengths (AB=107, BC=16, EF=184, FG=17), the collinearity of points D, E, C, F, and the concyclicity of points A, D, H, G. Emphasize the properties of rectangles (right angles, equal opposite sides) and the geometric implications of collinearity and concyclicity. Avoid making any assumptions about the orientation or relative placement of the rectangles at this stage."
    N_sc_1 = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc_1)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, listing given elements and constraints, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1, "Sub-task 1: Synthesize and choose the most consistent listing of given elements and constraints." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    debate_instruction_2 = "Sub-task 2: Clarify and verify all assumptions about the labeling and orientation of rectangles ABCD and EFGH, including the order of points D, E, C, F on the collinearity line and the order of points A, D, H, G on the circle. Explicitly avoid any unsupported assumptions such as fixing EF along the x-axis or assuming the collinearity line is horizontal. Instead, acknowledge that the orientation and relative placement are unknown and must be solved for. This subtask sets the foundation to prevent errors from premature coordinate fixing. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max_2)]
    all_answer2 = [[] for _ in range(N_max_2)]
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1], debate_instruction_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking1] + all_thinking2[r-1]
                thinking2, answer2 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, clarifying assumptions, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + all_thinking2[-1], "Sub-task 2: Synthesize and finalize assumptions about labeling and orientation." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3_1 = "Sub-task 3.1: Derive a coordinate system and vector representation for rectangle ABCD by placing point A at the origin and vector AB along the positive x-axis, using the given side lengths AB=107 and BC=16. This standard orientation is justified as a reference frame. Explicitly state that this choice is a fixed reference to simplify calculations and does not restrict the generality of the problem."
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", thinking2.content],
        "agent_collaboration": "CoT"
    }
    thinking3_1, answer3_1 = await cot_agent_3_1([taskInfo, thinking2], cot_instruction_3_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, deriving coordinate system for ABCD, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc3_1)
    print("Step 3.1: ", sub_tasks[-1])

    cot_instruction_3_2 = "Sub-task 3.2: Represent rectangle EFGH parametrically without fixing its orientation. Define vector EF as 184 times a unit vector u parameterized by an unknown angle theta (u = (cos theta, sin theta)). Define vector FG as 17 times the vector perpendicular to u. Express points E, F, G, and H in terms of theta and an unknown translation vector T. Avoid any assumptions that fix EF along a coordinate axis or fix the position of E arbitrarily. This parametrization preserves generality and respects the feedback to avoid unsupported assumptions."
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_3_2,
        "context": ["user query", thinking2.content],
        "agent_collaboration": "CoT"
    }
    thinking3_2, answer3_2 = await cot_agent_3_2([taskInfo, thinking2], cot_instruction_3_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_2.id}, parametrizing EFGH, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc3_2['response'] = {"thinking": thinking3_2, "answer": answer3_2}
    logs.append(subtask_desc3_2)
    print("Step 3.2: ", sub_tasks[-1])

    cot_sc_instruction_3_3 = "Sub-task 3.3: Express the collinearity condition of points D, E, C, F as a parametric linear equation. Since D and C belong to rectangle ABCD (coordinates known from subtask 3.1), and E and F belong to rectangle EFGH (parametrized in subtask 3.2), represent points E and C as points on the line through D and F with unknown scalar parameters t_E and t_C. Avoid fixing the line orientation or coordinates; instead, use vector equations to maintain generality. This subtask must explicitly avoid the previous error of assuming the collinearity line is horizontal or fixed in any coordinate direction."
    N_sc_3_3 = self.max_sc
    cot_agents_3_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc_3_3)]
    possible_answers_3_3 = []
    possible_thinkings_3_3 = []
    subtask_desc3_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_sc_instruction_3_3,
        "context": ["user query", thinking3_1.content, thinking3_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_3_3):
        thinking3_3, answer3_3 = await cot_agents_3_3[i]([taskInfo, thinking3_1, thinking3_2], cot_sc_instruction_3_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_3[i].id}, expressing collinearity condition, thinking: {thinking3_3.content}; answer: {answer3_3.content}")
        possible_answers_3_3.append(answer3_3)
        possible_thinkings_3_3.append(thinking3_3)
    final_decision_agent_3_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_3, answer3_3 = await final_decision_agent_3_3([taskInfo] + possible_thinkings_3_3, "Sub-task 3.3: Synthesize and finalize collinearity condition." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.3 output: thinking - {thinking3_3.content}; answer - {answer3_3.content}")
    subtask_desc3_3['response'] = {"thinking": thinking3_3, "answer": answer3_3}
    logs.append(subtask_desc3_3)
    print("Step 3.3: ", sub_tasks[-1])

    cot_instruction_3_4 = "Sub-task 3.4: Formulate the concyclicity condition for points A, D, H, G using their coordinate expressions. Use the general equation of a circle or the power of a point theorem to derive an equation involving theta, translation vector T, and other parameters from the previous subtasks. Ensure that the concyclicity condition is expressed symbolically and consistently with the parametrization of EFGH and ABCD. Avoid substituting any fixed coordinate values that contradict the general orientation assumptions."
    cot_agent_3_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3_4 = {
        "subtask_id": "stage_2.subtask_4",
        "instruction": cot_instruction_3_4,
        "context": ["user query", thinking3_1.content, thinking3_2.content],
        "agent_collaboration": "CoT"
    }
    thinking3_4, answer3_4 = await cot_agent_3_4([taskInfo, thinking3_1, thinking3_2], cot_instruction_3_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_4.id}, formulating concyclicity condition, thinking: {thinking3_4.content}; answer: {answer3_4.content}")
    sub_tasks.append(f"Sub-task 3.4 output: thinking - {thinking3_4.content}; answer - {answer3_4.content}")
    subtask_desc3_4['response'] = {"thinking": thinking3_4, "answer": answer3_4}
    logs.append(subtask_desc3_4)
    print("Step 3.4: ", sub_tasks[-1])

    debate_instruction_4_1 = "Sub-task 4.1: Solve the system of equations derived from the collinearity condition (subtask 3.3) and the concyclicity condition (subtask 3.4) simultaneously to find the unknown parameters theta, translation vector T, t_E, and t_C. Use symbolic or numeric methods as appropriate. This subtask must include iterative validation of assumptions and parameter values to ensure consistency with all given constraints and rectangle properties. Avoid premature fixing of parameters without justification. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_4_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_4_1 = self.max_round
    all_thinking4_1 = [[] for _ in range(N_max_4_1)]
    all_answer4_1 = [[] for _ in range(N_max_4_1)]
    subtask_desc4_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_4_1,
        "context": ["user query", thinking3_3.content, thinking3_4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4_1):
        for i, agent in enumerate(debate_agents_4_1):
            if r == 0:
                thinking4_1, answer4_1 = await agent([taskInfo, thinking3_3, thinking3_4], debate_instruction_4_1, r, is_sub_task=True)
            else:
                input_infos_4_1 = [taskInfo, thinking3_3, thinking3_4] + all_thinking4_1[r-1]
                thinking4_1, answer4_1 = await agent(input_infos_4_1, debate_instruction_4_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, solving system, thinking: {thinking4_1.content}; answer: {answer4_1.content}")
            all_thinking4_1[r].append(thinking4_1)
            all_answer4_1[r].append(answer4_1)
    final_decision_agent_4_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4_1, answer4_1 = await final_decision_agent_4_1([taskInfo] + all_thinking4_1[-1], "Sub-task 4.1: Synthesize and finalize solution for unknown parameters." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {thinking4_1.content}; answer - {answer4_1.content}")
    subtask_desc4_1['response'] = {"thinking": thinking4_1, "answer": answer4_1}
    logs.append(subtask_desc4_1)
    print("Step 4.1: ", sub_tasks[-1])

    reflect_inst_4_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_4_2 = "Sub-task 4.2: Validate the solution obtained in subtask 4.1 by checking that all given constraints are satisfied: rectangle side lengths, collinearity of D, E, C, F, and concyclicity of A, D, H, G. Confirm that no contradictions or unsupported assumptions have been introduced. If inconsistencies are found, iterate back to previous subtasks to refine parametrization or assumptions. This subtask ensures robustness and correctness of the solution." + reflect_inst_4_2
    cot_agent_4_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4_2 = self.max_round
    cot_inputs_4_2 = [taskInfo, thinking4_1]
    subtask_desc4_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": cot_reflect_instruction_4_2,
        "context": ["user query", thinking4_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking4_2, answer4_2 = await cot_agent_4_2(cot_inputs_4_2, cot_reflect_instruction_4_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4_2.id}, validating solution, thinking: {thinking4_2.content}; answer: {answer4_2.content}")
    critic_inst_4_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_4_2):
        feedback4_2, correct4_2 = await critic_agent_4_2([taskInfo, thinking4_2], critic_inst_4_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4_2.id}, providing feedback, thinking: {feedback4_2.content}; answer: {correct4_2.content}")
        if correct4_2.content == "True":
            break
        cot_inputs_4_2.extend([thinking4_2, feedback4_2])
        thinking4_2, answer4_2 = await cot_agent_4_2(cot_inputs_4_2, cot_reflect_instruction_4_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4_2.id}, refining validation, thinking: {thinking4_2.content}; answer: {answer4_2.content}")
    sub_tasks.append(f"Sub-task 4.2 output: thinking - {thinking4_2.content}; answer - {answer4_2.content}")
    subtask_desc4_2['response'] = {"thinking": thinking4_2, "answer": answer4_2}
    logs.append(subtask_desc4_2)
    print("Step 4.2: ", sub_tasks[-1])

    cot_instruction_5_1 = "Sub-task 5.1: Compute the length of segment CE using the parameters and coordinates obtained from the solved system. Since points C and E lie on the collinearity line parameterized by t_C and t_E, calculate CE as the absolute difference |t_C - t_E| multiplied by the norm of the direction vector of the line. Provide the final numeric value for CE. Ensure that this calculation is consistent with all previous constraints and validated parameters."
    cot_agent_5_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5_1 = {
        "subtask_id": "stage_4.subtask_1",
        "instruction": cot_instruction_5_1,
        "context": ["user query", thinking4_2.content],
        "agent_collaboration": "CoT"
    }
    thinking5_1, answer5_1 = await cot_agent_5_1([taskInfo, thinking4_2], cot_instruction_5_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5_1.id}, computing length CE, thinking: {thinking5_1.content}; answer: {answer5_1.content}")
    sub_tasks.append(f"Sub-task 5.1 output: thinking - {thinking5_1.content}; answer - {answer5_1.content}")
    subtask_desc5_1['response'] = {"thinking": thinking5_1, "answer": answer5_1}
    logs.append(subtask_desc5_1)
    print("Step 5.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5_1, answer5_1, sub_tasks, agents)
    return final_answer, logs
