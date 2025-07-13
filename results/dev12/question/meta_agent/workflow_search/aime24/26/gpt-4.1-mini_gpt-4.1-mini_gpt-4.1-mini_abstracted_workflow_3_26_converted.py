async def forward_26(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Compute the number of sets B for a given element a in A by deriving the formula for the count of finite nonempty subsets B with max(B) = a. "
        "Explain why this count is 2^(a-1), focusing on the combinatorial reasoning that elements less than a can be chosen arbitrarily while a must be included."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing count of sets B for element a, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    cot_instruction_2 = (
        "Sub-task 2: Express the total number of sets B as the sum over all a in A of 2^(a-1), and set this sum equal to 2024. "
        "Clarify that this transforms the problem into finding a subset A of positive integers such that the sum of 2^(a-1) over a in A equals 2024."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, expressing total sets B as sum of 2^(a-1), thinking: {thinking2.content}; answer: {answer2.content}")

    reflect_inst_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2 = "Sub-task 2: Your problem is to express total sets B as sum of 2^(a-1) equal to 2024." + reflect_inst_2
    cot_agent_reflect_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    cot_inputs_2 = [taskInfo, thinking1, answer1, thinking2, answer2]
    for i in range(N_max_2):
        feedback, correct = await critic_agent_2(cot_inputs_2, "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_reflect_2(cot_inputs_2, cot_reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflect_2.id}, refining expression of total sets B, thinking: {thinking2.content}; answer: {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append(subtask_desc2)

    cot_sc_instruction_3 = (
        "Sub-task 3: Formally represent the problem of finding the subset A as a problem of expressing 2024 as a sum of distinct powers of two of the form 2^(a-1). "
        "Emphasize the uniqueness of binary representation and the requirement that elements of A correspond to distinct exponents."
    )
    N_sc_3 = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    possible_answers_3 = []
    possible_thinkings_3 = []
    for i in range(N_sc_3):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, representing 2024 as sum of distinct powers of two, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_answers_3 + possible_thinkings_3, "Sub-task 3: Synthesize and choose the most consistent answer for representing 2024 as sum of distinct powers of two.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing representation of 2024, thinking: {thinking3.content}; answer: {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(subtask_desc3)

    cot_sc_instruction_4 = (
        "Sub-task 4: Decompose 2024 into a sum of distinct powers of two, identifying the exponents corresponding to elements of A. "
        "Carefully perform the binary decomposition of 2024 and map each power of two 2^(k) to an element a = k+1 in A."
    )
    N_sc_4 = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    possible_answers_4 = []
    possible_thinkings_4 = []
    for i in range(N_sc_4):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, decomposing 2024 into powers of two, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_answers_4 + possible_thinkings_4, "Sub-task 4: Synthesize and choose the most consistent decomposition of 2024.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing decomposition of 2024, thinking: {thinking4.content}; answer: {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(subtask_desc4)

    cot_sc_instruction_5 = (
        "Sub-task 5: Sum the elements of A identified from the decomposition in subtask_4 to find the final answer. "
        "Ensure the summation is accurate and clearly state the result."
    )
    N_sc_5 = self.max_sc
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "SC_CoT | Reflexion"
    }
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_sc_instruction_5, is_sub_task=True)
    agents.append(f"CoT-SC agent {cot_agent_5.id}, summing elements of A, thinking: {thinking5.content}; answer: {answer5.content}")

    reflect_inst_5 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_5 = "Sub-task 5: Your problem is to sum the elements of A from the decomposition." + reflect_inst_5
    cot_agent_reflect_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5(cot_inputs_5 + [thinking5, answer5], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_reflect_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflect_5.id}, refining sum of elements of A, thinking: {thinking5.content}; answer: {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    logs.append(subtask_desc5)

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
