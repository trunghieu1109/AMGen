async def forward_150(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: Normalize the state vector
    cot_sc_instruction = "Sub-task 1: Normalize the state vector (-1, 2, 1)ᵀ by computing its Euclidean norm and dividing each component by that norm."
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents:
        thinking1, answer1 = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings.append(thinking1)
        possible_answers.append(answer1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings + possible_answers, "Sub-task 1: Synthesize and output the normalized state vector.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Determine the projector onto the zero eigenspace of P
    cot_sc_instruction = "Sub-task 2: Solve P v = 0 for v, find an orthonormal basis of the nullspace, and construct the projector Π₀ = Σ |v⟩⟨v|."
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents:
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings.append(thinking2)
        possible_answers.append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings + possible_answers, "Sub-task 2: Synthesize and output the projector Π₀.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Compute the probability p(0) = ⟨ψ|Π₀|ψ⟩
    cot_sc_instruction = "Sub-task 3: Using the normalized state and Π₀, compute p(0) = ⟨ψ_normalized| Π₀ |ψ_normalized⟩."
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents:
        thinking3, answer3 = await agent([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_thinkings.append(thinking3)
        possible_answers.append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking1, answer1, thinking2, answer2] + possible_thinkings + possible_answers, "Sub-task 3: Synthesize and output the numerical probability p(0).", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Compare with provided choices and select the correct one using Reflexion
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 4: Compare the computed p(0) with the provided choices (1/3, 2/3, √(2/3), 1) and select the correct one. " + reflect_inst
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs = [taskInfo, thinking3, answer3]
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_reflect_instruction, "context": ["user query", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "Reflexion"}
    thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking4, answer4], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs