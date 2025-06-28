async def forward_22(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Stage 1: Condition Identification and Modular Reasoning

    # Sub-task 1: Determine the potential number of elements in the list that would sum to 30 and could have a mode
    cot_instruction_1 = "Sub-task 1: Determine the potential number of elements in the list that would sum to 30 and could have a mode"
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculating potential list configurations, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Identify the frequency of 9 in the list, given that it is the unique mode
    cot_sc_instruction_2 = "Sub-task 2: Based on the output from Sub-task 1, identify the frequency of 9 in the list"
    N_sc_2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "CoT-SC Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}

    for i in range(N_sc_2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining frequency of 9, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    answer2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2]
    answer2 = answermapping_2[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Construct a list of integers totaling 30, including multiple occurrences of 9, while considering the median condition.
    cot_reflect_instruction_3 = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, construct a list while considering median condition"
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "CoT Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round

    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, constructing list, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max_3):
        feedback3, correct3 = await critic_agent_3([taskInfo, thinking3, answer3], "please review the list construction and provide feedback about limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining list, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Intermediate Inference and Validation

    # Sub-task 4: Check the possible configurations of the list from subtask 3 to ensure the median is an integer not in the list and they adjust accordingly.
    cot_reflect_instruction_4 = "Sub-task 4: Check possible list configurations for median validity and adjust if needed"
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "CoT Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round

    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, checking and adjusting list, thinking: {thinking4.content}; answer: {answer4.content}")

    for i in range(N_max_4):
        feedback4, correct4 = await critic_agent_4([taskInfo, thinking4, answer4], "review list configurations for median validity.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback4.content}; answer: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, fine-tuning list, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Validate the final list for all given conditions: total sum, unique mode, and median restrictions.
    cot_reflect_instruction_5 = "Sub-task 5: Validate the final list for all conditions."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "CoT Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round

    cot_inputs_5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, validating final list, thinking: {thinking5.content}; answer: {answer5.content}")

    for i in range(N_max_5):
        feedback5, correct5 = await critic_agent_5([taskInfo, thinking5, answer5], "review final list to ensure it meets all criteria.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback5.content}; answer: {correct5.content}")
        if correct5.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, finalizing validation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Stage 3: Final Output Generation and Integration

    # Sub-task 6: Compute the sum of squares of numbers from the validated list.
    cot_reflect_instruction_6 = "Sub-task 6: Compute the sum of squares of numbers from the validated list"
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "CoT Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_reflect_instruction_6, is_sub_task=True)
    agents.append(f"Final CoT agent {cot_agent_6.id}, computing sum of squares, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer