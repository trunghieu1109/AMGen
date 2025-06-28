async def forward_125(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze Reaction I and Reaction II independently using Self-Consistency Chain-of-Thought

    cot_sc_instruction_1 = (
        "Sub-task 1: Analyze Reaction I: Determine the product(s) formed when (S)-5-methoxyhexan-3-one is treated with LAH followed by acidic workup, "
        "including stereochemical outcomes and number of distinct products formed."
    )
    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze Reaction II: Determine the product(s) formed when pentane-2,4-dione is treated with excess NaBH4 followed by acidic workup, "
        "including stereochemical outcomes and number of distinct products formed."
    )

    N_sc = self.max_sc if hasattr(self, 'max_sc') else 5

    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]

    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}

    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}

    for i in range(N_sc):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, analyzing Reaction I, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1.content)
        thinkingmapping_1[answer1.content] = thinking1
        answermapping_1[answer1.content] = answer1

        thinking2, answer2 = await cot_agents_2[i]([taskInfo], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing Reaction II, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    # Select the most consistent answers for Reaction I and II (majority vote)
    counter_1 = Counter(possible_answers_1)
    best_answer_1 = counter_1.most_common(1)[0][0]
    thinking1 = thinkingmapping_1[best_answer_1]
    answer1 = answermapping_1[best_answer_1]

    counter_2 = Counter(possible_answers_2)
    best_answer_2 = counter_2.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]

    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Combine products and predict chromatographic behavior

    # Sub-task 3: Combine products and identify total distinct species (Reflexion)
    cot_reflect_instruction_3 = (
        "Sub-task 3: Combine the products from Reaction I and Reaction II and identify the total number of distinct chemical species present in the mixture, "
        "considering stereoisomers and structural isomers."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round if hasattr(self, 'max_round') else 3

    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]

    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, combining products and identifying distinct species, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                                "Review the combined product species identification for completeness and correctness.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback on combined species, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining combined species identification, thinking: {thinking3.content}; answer: {answer3.content}")

    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Predict chromatographic behavior on normal-phase HPLC (Chain-of-Thought)
    cot_instruction_4 = (
        "Sub-task 4: Predict the chromatographic behavior of the combined product mixture on a normal-phase HPLC column, "
        "including the expected number of peaks based on chemical and stereochemical differences."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, predicting normal-phase HPLC peaks, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Predict chromatographic behavior on chiral stationary phase HPLC (Chain-of-Thought)
    cot_instruction_5 = (
        "Sub-task 5: Predict the chromatographic behavior of the combined product mixture on a chiral stationary phase HPLC column, "
        "including the expected number of peaks based on enantiomeric and diastereomeric resolution."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, predicting chiral HPLC peaks, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Stage 3: Integrate chromatographic predictions and select correct answer

    # Sub-task 6: Integrate predicted peak counts from normal-phase and chiral HPLC
    cot_instruction_6 = (
        "Sub-task 6: Integrate the predicted peak counts from normal-phase and chiral HPLC chromatograms to determine the total number of peaks observed in each chromatogram after combining products from both reactions."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking4, answer4, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, integrating chromatographic peak predictions, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Sub-task 7: Compare integrated predictions with multiple-choice options using Debate
    debate_instruction_7 = (
        "Sub-task 7: Compare the integrated chromatographic peak predictions with the provided multiple-choice options to select the correct answer regarding the number of peaks observed in both chromatograms."
    )
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round if hasattr(self, 'max_round') else 3
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]

    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct multiple-choice answer, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)

    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the correct multiple-choice answer.", is_sub_task=True)
    agents.append(f"Final Decision agent on selecting correct answer, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
