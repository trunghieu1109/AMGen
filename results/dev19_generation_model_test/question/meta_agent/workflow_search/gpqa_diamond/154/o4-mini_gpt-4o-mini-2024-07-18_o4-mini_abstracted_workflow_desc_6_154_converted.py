async def forward_154(self, taskInfo):
    from collections import Counter
    print("Task Requirement:", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0, subtask_0_1: SC-CoT to extract matrices and state
    cot_sc_instruction0 = "Sub-task 0_1: Extract and summarize the matrix representations of Px, Py, Pz and the system state vector components in the Pz basis for the given problem."
    N0 = self.max_sc
    cot_agents0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N0)]
    possible_thinkings0 = []
    possible_answers0 = []
    subtask_desc0 = {"subtask_id": "subtask_0_1", "instruction": cot_sc_instruction0, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N0):
        thinking0, answer0 = await cot_agents0[i]([taskInfo], cot_sc_instruction0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents0[i].id}, analyzing operators and state, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_thinkings0.append(thinking0)
        possible_answers0.append(answer0)
    final_decision0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    decision_instr0 = "Sub-task 0_1: Given all the above thinking and answers, synthesize the consistent and correct summary of the matrices Px, Py, Pz and the state vector."
    thinking0_1, answer0_1 = await final_decision0([taskInfo] + possible_thinkings0 + possible_answers0, decision_instr0, is_sub_task=True)
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking0_1.content}; answer - {answer0_1.content}")
    subtask_desc0['response'] = {"thinking": thinking0_1, "answer": answer0_1}
    logs.append(subtask_desc0)
    print("Step 1:", sub_tasks[-1])

    # Stage 1, subtask_1_1: SC-CoT to compute expectation ⟨Pz⟩
    cot_sc_instruction1 = "Sub-task 1_1: Compute the expectation value of Pz by applying Pz to the state vector and taking the inner product."
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id": "subtask_1_1", "instruction": cot_sc_instruction1, "context": ["user query", thinking0_1.content, answer0_1.content], "agent_collaboration": "SC_CoT"}
    for i in range(N1):
        thinking1, answer1 = await cot_agents1[i]([taskInfo, thinking0_1, answer0_1], cot_sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, computing <Pz>, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings1.append(thinking1)
        possible_answers1.append(answer1)
    final_decision1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    decision_instr1 = "Sub-task 1_1: Given all the above thinking and answers, determine the consistent expectation value of Pz."
    thinking1_1, answer1_1 = await final_decision1([taskInfo, thinking0_1, answer0_1] + possible_thinkings1 + possible_answers1, decision_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc1)
    print("Step 2:", sub_tasks[-1])

    # Stage 2, subtask_2_1: Reflexion to compute ΔPz and match choice
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction2 = "Sub-task 2_1: Calculate the uncertainty ΔPz = sqrt(<Pz^2> - <Pz>^2) and match it to one of the given choices." + reflect_inst
    critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    max_round = self.max_round
    inputs2 = [taskInfo, thinking1_1, answer1_1]
    subtask_desc2 = {"subtask_id": "subtask_2_1", "instruction": cot_reflect_instruction2, "context": ["user query", thinking1_1.content, answer1_1.content], "agent_collaboration": "Reflexion"}
    thinking2, answer2 = await cot_agent2(inputs2, cot_reflect_instruction2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent2.id}, initial uncertainty calc, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(max_round):
        feedback2, correct2 = await critic_agent2([taskInfo, thinking2, answer2], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent2.id}, feedback: {feedback2.content}; correct: {correct2.content}")
        if correct2.content.strip() == "True":
            break
        inputs2.extend([thinking2, answer2, feedback2])
        thinking2, answer2 = await cot_agent2(inputs2, cot_reflect_instruction2, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent2.id}, refined uncertainty calc, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 3:", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2, answer2, sub_tasks, agents)
    return final_answer, logs