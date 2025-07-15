async def forward_187(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = (
        "Sub-task 1: Derive the general formula for the interplanar distance d_hkl in a rhombohedral lattice given lattice parameter a and angle alpha, "
        "and Miller indices (hkl). Provide detailed step-by-step reasoning and the final formula."
    )
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_sc_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, deriving formula, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings_1.append(thinking1)
        possible_answers_1.append(answer1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1 + possible_answers_1, 
                                                    "Sub-task 1: Synthesize and choose the most consistent and correct formula for interplanar distance in rhombohedral lattice.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    reflect_instruction_2 = (
        "Sub-task 2: Clarify and confirm the interpretation of the given parameters: that the interatomic distance corresponds to the rhombohedral lattice parameter a = 10 Angstrom, "
        "and that the angles alpha=beta=gamma=30 degrees are internal lattice angles. Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2 = [taskInfo]
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": reflect_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, clarifying parameters, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        critic_inst_2 = (
            "Please review the answer above and criticize on where might be wrong. "
            "If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
        )
        feedback, correct = await critic_agent_2([taskInfo, thinking2, answer2], critic_inst_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining parameter clarification, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Apply the derived formula from Sub-task 1 to the specific case of the (111) plane with lattice parameter a=10 Angstrom and alpha=30 degrees. "
        "Calculate the numerical value of the interplanar distance d_111 with detailed step-by-step reasoning."
    )
    cot_sc_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking1, answer1, thinking2, answer2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking3, answer3 = await cot_sc_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3[i].id}, calculating d_111, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_thinkings_3.append(thinking3)
        possible_answers_3.append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking1, answer1, thinking2, answer2] + possible_thinkings_3 + possible_answers_3, 
                                                    "Sub-task 3: Synthesize and choose the most consistent and correct numerical value for d_111.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    reflect_instruction_4 = (
        "Sub-task 4: Evaluate the calculated interplanar distance from Sub-task 3 against the given choices (9.54, 8.95, 9.08, 10.05 Angstrom). "
        "Select the closest matching value with justification. Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": reflect_instruction_4,
        "context": ["user query", thinking3, answer3],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, evaluating choices, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        critic_inst_4 = (
            "Please review the answer above and criticize on where might be wrong. "
            "If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
        )
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], critic_inst_4, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining evaluation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_5 = (
        "Sub-task 5: Summarize the final answer with justification based on the calculation and choice comparison from Sub-task 4, "
        "ensuring clarity and correctness."
    )
    cot_sc_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_5 = []
    possible_thinkings_5 = []
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", thinking4, answer4],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking5, answer5 = await cot_sc_agents_5[i]([taskInfo, thinking4, answer4], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_5[i].id}, summarizing final answer, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_thinkings_5.append(thinking5)
        possible_answers_5.append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo, thinking4, answer4] + possible_thinkings_5 + possible_answers_5, 
                                                    "Sub-task 5: Synthesize and provide the final answer with justification.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc_5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
