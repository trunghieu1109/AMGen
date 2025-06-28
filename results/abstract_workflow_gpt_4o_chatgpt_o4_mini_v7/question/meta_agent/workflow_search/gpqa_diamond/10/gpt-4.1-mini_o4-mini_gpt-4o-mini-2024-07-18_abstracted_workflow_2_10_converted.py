async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and classify molecular biology concepts in each choice
    cot_instruction_1 = (
        "Sub-task 1: Analyze and classify the molecular biology concepts and mechanisms mentioned in each choice related to SARS-CoV-2, "
        "including programmed ribosomal frameshifting, nsp10/nsp14-ExoN complex function, ORF3a-induced apoptosis pathway, and frameshifting pseudoknot conformations. "
        "Identify key attributes, relationships, and functions described in each statement."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing molecular biology concepts in choices, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Stage 1 subtasks 2-5: Extract detailed features for each concept using Self-Consistency Chain-of-Thought
    N_sc = self.max_sc if hasattr(self, 'max_sc') else 5

    # Sub-task 2: Extract features about programmed ribosomal frameshifting
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on Sub-task 1 classification, extract detailed features and known scientific facts about SARS-CoV-2 programmed ribosomal frameshifting, "
        "including the role of slippery sequences, pseudoknots, and comparison with SARS-CoV frameshifting conformation."
    )
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, extracting features of programmed ribosomal frameshifting, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    counter_2 = Counter(possible_answers_2)
    answer2_final = counter_2.most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_final]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Extract features about nsp10/nsp14-ExoN complex
    cot_sc_instruction_3 = (
        "Sub-task 3: Based on Sub-task 1 classification, extract detailed features and known scientific facts about the nsp10/nsp14-ExoN heterodimer complex in SARS-CoV-2, "
        "focusing on its mismatch repair mechanism, exonuclease activity, and interaction preventing dsRNA breakdown."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, extracting features of nsp10/nsp14-ExoN complex, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    counter_3 = Counter(possible_answers_3)
    answer3_final = counter_3.most_common(1)[0][0]
    thinking3_final = thinkingmapping_3[answer3_final]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Extract features about ORF3a-induced apoptosis pathway
    cot_sc_instruction_4 = (
        "Sub-task 4: Based on Sub-task 1 classification, extract detailed features and known scientific facts about SARS-CoV-2 ORF3a protein's role in apoptosis, "
        "specifically its ability to activate caspase-8 without affecting Bcl-2 expression, and the implication of extrinsic versus mitochondrial apoptotic pathways."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking1, answer1], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, extracting features of ORF3a apoptosis pathway, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    counter_4 = Counter(possible_answers_4)
    answer4_final = counter_4.most_common(1)[0][0]
    thinking4_final = thinkingmapping_4[answer4_final]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_final.content}; answer - {answer4_final}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Extract features about frameshifting rate and pseudoknot conformations
    cot_sc_instruction_5 = (
        "Sub-task 5: Based on Sub-task 1 classification, extract detailed features and known scientific facts about the relationship between frameshifting rate and pseudoknot conformations in SARS-CoV and SARS-CoV-2, "
        "including the number of conformations under tension and their correlation with frameshifting efficiency."
    )
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_5 = []
    thinkingmapping_5 = {}
    answermapping_5 = {}
    for i in range(N_sc):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking1, answer1], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, extracting features of frameshifting rate and pseudoknot conformations, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5.content)
        thinkingmapping_5[answer5.content] = thinking5
        answermapping_5[answer5.content] = answer5
    counter_5 = Counter(possible_answers_5)
    answer5_final = counter_5.most_common(1)[0][0]
    thinking5_final = thinkingmapping_5[answer5_final]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5_final.content}; answer - {answer5_final}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Stage 2: Evaluate accuracy of each choice statement
    # Sub-task 6: Evaluate choice1 about programmed ribosomal frameshifting
    cot_instruction_6 = (
        "Sub-task 6: Evaluate the accuracy and correctness of the statement in choice1 by comparing extracted features about programmed ribosomal frameshifting in SARS-CoV-2 and SARS-CoV from Sub-task 2 with established scientific knowledge."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking2_final, answer2_final], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, evaluating choice1 correctness, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Sub-task 7: Evaluate choice2 about nsp10/nsp14-ExoN complex
    cot_instruction_7 = (
        "Sub-task 7: Evaluate the accuracy and correctness of the statement in choice2 by comparing extracted features about nsp10/nsp14-ExoN complex function from Sub-task 3 with established scientific knowledge."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking3_final, answer3_final], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, evaluating choice2 correctness, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    # Sub-task 8: Evaluate choice3 about ORF3a-induced apoptosis pathway
    cot_instruction_8 = (
        "Sub-task 8: Evaluate the accuracy and correctness of the statement in choice3 by comparing extracted features about ORF3a-induced apoptosis pathway from Sub-task 4 with established scientific knowledge."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking4_final, answer4_final], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, evaluating choice3 correctness, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Subtask 8 answer: ", sub_tasks[-1])

    # Sub-task 9: Evaluate choice4 about frameshifting rate and pseudoknot conformations
    cot_instruction_9 = (
        "Sub-task 9: Evaluate the accuracy and correctness of the statement in choice4 by comparing extracted features about frameshifting rate and pseudoknot conformations from Sub-task 5 with established scientific knowledge."
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking5_final, answer5_final], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, evaluating choice4 correctness, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Subtask 9 answer: ", sub_tasks[-1])

    # Sub-task 10: Compare evaluation results to identify the incorrect statement
    debate_roles = ["Pro", "Con"]
    debate_agents_10 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_10 = self.max_round if hasattr(self, 'max_round') else 3
    all_thinking_10 = [[] for _ in range(N_max_10)]
    all_answer_10 = [[] for _ in range(N_max_10)]

    debate_instruction_10 = (
        "Sub-task 10: Compare the evaluation results of all four choices (Sub-tasks 6 to 9) to identify which statement is incorrect, "
        "thereby answering the original question about which statement is false regarding the molecular biology of SARS-CoV-2."
    )

    for r in range(N_max_10):
        for i, agent in enumerate(debate_agents_10):
            if r == 0:
                thinking10, answer10 = await agent(
                    [taskInfo, thinking6, answer6, thinking7, answer7, thinking8, answer8, thinking9, answer9],
                    debate_instruction_10, r, is_sub_task=True)
            else:
                input_infos_10 = [taskInfo, thinking6, answer6, thinking7, answer7, thinking8, answer8, thinking9, answer9] + all_thinking_10[r-1] + all_answer_10[r-1]
                thinking10, answer10 = await agent(input_infos_10, debate_instruction_10, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating incorrect statement, thinking: {thinking10.content}; answer: {answer10.content}")
            all_thinking_10[r].append(thinking10)
            all_answer_10[r].append(answer10)

    final_decision_agent_10 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking10_final, answer10_final = await final_decision_agent_10(
        [taskInfo] + all_thinking_10[-1] + all_answer_10[-1],
        "Sub-task 10: Make final decision on which statement is incorrect among the four choices.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, deciding incorrect statement, thinking: {thinking10_final.content}; answer: {answer10_final.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10_final.content}; answer - {answer10_final.content}")
    print("Subtask 10 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking10_final, answer10_final, sub_tasks, agents)
    return final_answer
