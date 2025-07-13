async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Problem Definition and Formulation
    cot_instruction_0_1 = (
        "Sub-task 0_1: Precisely define the domain of the problem by characterizing all two-digit numbers n in base b, "
        "where b >= 2, with digits x and y satisfying 1 <= x <= b-1 and 0 <= y <= b-1. Emphasize the positional representation and digit constraints without attempting to solve the equation."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "subtask_0_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, defining problem domain, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0_1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 0_2: Formulate the key equation relating digits and n: x*b + y = (x + y)^2, and clarify the constraints on x, y, and b, "
        "emphasizing that sqrt(n) must be an integer and n must be a perfect square. Avoid solving the equation here; focus on precise formulation and implications."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "subtask_0_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, formulating key equation, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0_2: ", sub_tasks[-1])

    cot_instruction_0_3 = (
        "Sub-task 0_3: Analyze the implications of the digit sum condition x + y = sqrt(n) on the possible values of x, y, and b, "
        "including bounds on the digit sum and the range of n. Discuss how these constraints limit the search space for valid (x,y) pairs for each base b."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "subtask_0_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_2], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, analyzing digit sum implications, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0_3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0_3: ", sub_tasks[-1])

    # Stage 1: Enumeration and Counting
    cot_sc_instruction_1_1 = (
        "Sub-task 1_1: Develop a systematic method to enumerate all pairs (x, y) for a fixed base b that satisfy the equation x*b + y = (x + y)^2 with digit constraints, "
        "ensuring n is two-digit in base b and a perfect square. The method should be explicit and algorithmic, suitable for implementation or simulation."
    )
    cot_sc_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_1_1, answer_1_1 = await cot_sc_agents_1_1[i]([taskInfo, thinking_0_3], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_1[i].id}, enumerating method, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_thinkings_1_1, "Sub-task 1_1: Synthesize and choose the most consistent enumeration method.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_3.content],
        "agent_collaboration": "SC_CoT",
        "response": {"thinking": thinking_1_1, "answer": answer_1_1}
    }
    logs.append(subtask_desc_1_1)
    print("Step 1_1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 1_2: Implement or simulate the enumeration method for increasing values of b starting from 2, explicitly counting the number of b-eautiful integers for each base b. "
        "Produce a detailed table listing each base b alongside the count of valid (x,y) pairs and corresponding b-eautiful integers. Continue enumeration until the count exceeds ten for some base b."
    )
    cot_sc_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "subtask_1_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_1_2, answer_1_2 = await cot_sc_agents_1_2[i]([taskInfo, thinking_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_2[i].id}, enumerating counts per base, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 1_2: Synthesize and choose the most consistent enumeration counts and table.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2 = {
        "subtask_id": "subtask_1_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "SC_CoT",
        "response": {"thinking": thinking_1_2, "answer": answer_1_2}
    }
    logs.append(subtask_desc_1_2)
    print("Step 1_2: ", sub_tasks[-1])

    cot_reflect_instruction_1_3 = (
        "Sub-task 1_3: Perform a focused verification and cross-check of the enumeration results for bases near the suspected threshold (e.g., b=10 to b=20). "
        "Confirm that all valid pairs are included, digit constraints are strictly enforced, and no duplicates or omissions occur. Document any corrections or confirmations in the enumeration table. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_3 = self.max_round
    cot_inputs_1_3 = [taskInfo, thinking_1_2, answer_1_2]
    subtask_desc_1_3 = {
        "subtask_id": "subtask_1_3",
        "instruction": cot_reflect_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, verifying enumeration near threshold, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    for i in range(N_max_1_3):
        feedback_1_3, correct_1_3 = await critic_agent_1_3([taskInfo, thinking_1_3], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, providing feedback, thinking: {feedback_1_3.content}; answer: {correct_1_3.content}")
        if correct_1_3.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, feedback_1_3])
        thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining verification, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1_3: ", sub_tasks[-1])

    # Stage 2: Final Analysis and Conclusion
    cot_reflect_instruction_2_1 = (
        "Sub-task 2_1: Analyze the verified enumeration data to identify the least integer base b >= 2 for which the count of b-eautiful integers exceeds ten. "
        "Provide a clear justification based on the explicit enumeration table, ensuring the conclusion is fully data-driven and supported by the verified counts. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_1 = self.max_round
    cot_inputs_2_1 = [taskInfo, thinking_1_3, answer_1_3]
    subtask_desc_2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, analyzing enumeration data for final answer, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(N_max_2_1):
        feedback_2_1, correct_2_1 = await critic_agent_2_1([taskInfo, thinking_2_1], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback_2_1.content}; answer: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, feedback_2_1])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining final analysis, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2_1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs
