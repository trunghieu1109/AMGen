async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0 = (
        "Sub-task 1: Derive and validate the formal mathematical representation of b-eautiful numbers. "
        "Explicitly state digit constraints and the key equation relating digits, base, and perfect square condition. "
        "Ensure clarity on digit ranges and two-digit number condition."
    )
    cot_sc_agents_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_0 = []
    possible_thinkings_0 = []
    subtask_desc0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking0, answer0 = await cot_sc_agents_0[i]([taskInfo], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0[i].id}, deriving formal representation, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_answers_0.append(answer0)
        possible_thinkings_0.append(thinking0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_thinkings_0 + possible_answers_0, "Sub-task 1: Synthesize and validate the formal mathematical representation of b-eautiful numbers.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 1: ", sub_tasks[-1])

    reflect_inst_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1 = (
        "Sub-task 1: Transform the key equation into an analyzable form by expressing n = d1*b + d0 and s = d1 + d0, "
        "relate these to the perfect square condition n = s^2 under digit constraints. Clarify the relationship and prepare for enumeration. "
        + reflect_inst_1
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_1 = [taskInfo, thinking0, answer0]
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_reflect_instruction_1,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        "agent_collaboration": "Reflexion"
    }
    thinking1, answer1 = await cot_agent_1(cot_inputs_1, cot_reflect_instruction_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1.id}, transforming key equation, thinking: {thinking1.content}; answer: {answer1.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_1([taskInfo, thinking1, answer1], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_1.extend([thinking1, answer1, feedback])
        thinking1, answer1 = await cot_agent_1(cot_inputs_1, cot_reflect_instruction_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1.id}, refining transformation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 1: Develop and implement a complete enumeration method to find all valid digit pairs (d1, d0) for each base b >= 2 that satisfy d1*b + d0 = (d1 + d0)^2 with digit constraints. "
        "Systematically iterate over all possible digit pairs and bases, ensuring no pairs are missed or double counted."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "agent_collaboration": "CoT"
    }
    thinking2_1, answer2_1 = await cot_agent_2_1([taskInfo, thinking1, answer1], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, enumerating valid digit pairs, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc2_1)
    print("Step 3.1: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = (
        "Sub-task 2: Verify and cross-validate the enumeration results for each base b by producing explicit tables or lists of valid pairs and their counts. "
        "Reconcile any conflicting counts and ensure accuracy before proceeding."
    )
    cot_sc_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2_2, answer2_2 = await cot_sc_agents_2_2[i]([taskInfo, thinking2_1, answer2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2_2[i].id}, verifying enumeration, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
        possible_answers_2_2.append(answer2_2)
        possible_thinkings_2_2.append(thinking2_2)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_2, answer2_2 = await final_decision_agent_2_2([taskInfo, thinking2_1, answer2_1] + possible_thinkings_2_2 + possible_answers_2_2, "Sub-task 3.2: Synthesize and verify enumeration results.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc2_2['response'] = {"thinking": thinking2_2, "answer": answer2_2}
    logs.append(subtask_desc2_2)
    print("Step 3.2: ", sub_tasks[-1])

    reflect_inst_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_3 = (
        "Sub-task 1: Iterate over bases b >= 2 using the verified enumeration data to identify the smallest base b for which the count of b-eautiful numbers exceeds 10. "
        "Explicitly reference the verified counts and include a reconciliation process to confirm the minimal base without ambiguity or conflicting claims. "
        + reflect_inst_3
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_3 = [taskInfo, thinking2_2, answer2_2]
    subtask_desc3 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, identifying minimal base, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining minimal base identification, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
