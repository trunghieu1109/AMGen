async def forward_154(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Extract and summarize the matrix forms of P_x, P_y, P_z, the eigenbasis and eigenvalues of P_z, the explicit state vector ψ, and the target quantity ΔP_z."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, summarizing inputs, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Confirm that the state vector ψ = (-1/2, 1/√2, -1/2)^T is normalized by calculating its norm squared."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, checking normalization, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings.append(thinking2_i)
        possible_answers.append(answer2_i)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings + possible_answers, "Sub-task 2: Synthesize and choose the most consistent confirmation for normalization.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 3: Compute the expectation value <P_z> = ψ† P_z ψ by using the state probabilities and eigenvalues."
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking3_i, answer3_i = await cot_agents[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, computing <P_z>, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_thinkings.append(thinking3_i)
        possible_answers.append(answer3_i)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking1, answer1, thinking2, answer2] + possible_thinkings + possible_answers, "Sub-task 3: Synthesize and choose the most consistent value for <P_z>.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 4: Compute the expectation value <P_z^2> = ψ† P_z^2 ψ by squaring the eigenvalues and applying the state probabilities."
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking4_i, answer4_i = await cot_agents[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, computing <P_z^2>, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_thinkings.append(thinking4_i)
        possible_answers.append(answer4_i)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking1, answer1, thinking2, answer2] + possible_thinkings + possible_answers, "Sub-task 4: Synthesize and choose the most consistent value for <P_z^2>.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 5: Calculate ΔP_z = sqrt(<P_z^2> - <P_z>^2) and select the matching answer from the provided choices." + reflect_inst
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking3, answer3, thinking4, answer4]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_reflect_instruction, "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "Reflexion"}
    thinking5, answer5 = await cot_agent5(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, initial calculation of ΔP_z, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking5, answer5], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent5(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, refinement, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs