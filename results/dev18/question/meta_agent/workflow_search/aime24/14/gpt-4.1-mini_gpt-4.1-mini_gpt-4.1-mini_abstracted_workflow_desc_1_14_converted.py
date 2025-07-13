async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = "Sub-task 1: Parametrize the points A, B, C, D on the hyperbola using vectors representing the diagonals AC and BD, considering that the diagonals intersect at the origin and that A and C are symmetric about the origin, as are B and D."
    N = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N)]
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1 = []
    possible_thinkings_1 = []
    for i in range(N):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, parametrize points, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1, "Sub-task 1: Synthesize and choose the most consistent parametrization for points A, B, C, D on the hyperbola.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Express the coordinates of points A, B, C, D in terms of the diagonal vectors and write down the hyperbola equation constraints for these points, based on the parametrization from Sub-task 1."
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2 = []
    possible_thinkings_2 = []
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, express coordinates and hyperbola constraints, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_thinkings_2, "Sub-task 2: Synthesize and choose the most consistent expressions and hyperbola constraints.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = "Sub-task 3: Formulate the rhombus conditions: equal side lengths and perpendicular diagonals, translating these geometric properties into algebraic equations involving the diagonal vectors, based on the parametrization from Sub-task 1."
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_3 = []
    possible_thinkings_3 = []
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, formulate rhombus conditions, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_thinkings_3, "Sub-task 3: Synthesize and choose the most consistent rhombus conditions.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = "Sub-task 4: Combine the hyperbola constraints and rhombus conditions to derive a system of equations relating the diagonal vectors, ensuring all points lie on the hyperbola and the figure is a rhombus with diagonals intersecting at the origin."
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking2.content, thinking3.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_4 = []
    possible_thinkings_4 = []
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2, thinking3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, combine constraints, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_thinkings_4, "Sub-task 4: Synthesize and choose the most consistent system of equations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_5 = "Sub-task 5: Derive an expression for BD^2 in terms of the diagonal vectors and simplify it using the system of equations obtained from the constraints in Sub-task 4."
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_5 = []
    possible_thinkings_5 = []
    for i in range(N):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking4], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, derive BD^2 expression, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5)
        possible_thinkings_5.append(thinking5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + possible_thinkings_5, "Sub-task 5: Synthesize and choose the most consistent expression for BD^2.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_6 = "Sub-task 6: Set up the optimization problem to find the supremum of BD^2 under the derived constraints, identifying the variables and parameters to optimize over, based on the expression from Sub-task 5."
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_6,
        "context": ["user query", thinking5.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_6 = []
    possible_thinkings_6 = []
    for i in range(N):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, set up optimization problem, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6)
        possible_thinkings_6.append(thinking6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + possible_thinkings_6, "Sub-task 6: Synthesize and choose the most consistent optimization setup.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_sc_instruction_7 = "Sub-task 7: Solve the optimization problem analytically or by suitable methods to find the greatest real number less than BD^2 for all such rhombi, and verify the solution satisfies all constraints."
    cot_agents_7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N)]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_sc_instruction_7,
        "context": ["user query", thinking6.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_7 = []
    possible_thinkings_7 = []
    for i in range(N):
        thinking7, answer7 = await cot_agents_7[i]([taskInfo, thinking6], cot_sc_instruction_7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_7[i].id}, solve optimization, thinking: {thinking7.content}; answer: {answer7.content}")
        possible_answers_7.append(answer7)
        possible_thinkings_7.append(thinking7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + possible_thinkings_7, "Sub-task 7: Synthesize and choose the most consistent final solution.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
