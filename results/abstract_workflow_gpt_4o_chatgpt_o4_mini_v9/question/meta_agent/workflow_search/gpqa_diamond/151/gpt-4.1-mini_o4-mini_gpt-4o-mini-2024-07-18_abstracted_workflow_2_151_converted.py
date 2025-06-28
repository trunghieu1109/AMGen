async def forward_151(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements

    # Sub-task 1: Analyze the biological context of the quorum-sensing peptide and its effect on yeast shmoo formation using Chain-of-Thought
    cot_instruction_1 = (
        "Sub-task 1: Analyze the biological context of the quorum-sensing peptide identified in Lactobacillus acidophilus and its observed effect on Saccharomyces cerevisiae, "
        "specifically the induction of shmoo formation, to understand the relevance of intercellular communication mechanisms across species."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing quorum-sensing peptide effect, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Classify and define protein complexes involved in chromatin biology using Self-Consistency Chain-of-Thought
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the analysis from Sub-task 1, classify and define the different protein complexes involved in chromatin biology and gene regulation in eukaryotes, "
        "focusing on the pre-initiation complex, pre-replication complex, enhancer protein complex, and nucleosome histone complex, to establish their roles and presence in active chromatin regions."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, classifying protein complexes, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer by frequency
    answer2_counter = Counter(possible_answers_2)
    most_common_answer2 = answer2_counter.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer2]
    answer2 = answermapping_2[most_common_answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Evaluate and Select Elements

    # Sub-task 3: Evaluate which protein complexes are associated with active chromatin during transcriptional activation using Reflexion
    cot_reflect_instruction_3 = (
        "Sub-task 3: Evaluate which of the classified protein complexes (from Sub-task 2) are typically associated with active chromatin regions during transcriptional activation, "
        "especially in the context of shmoo formation in yeast, to predict their likelihood of being detected by chromatin immunoprecipitation followed by mass spectrometry."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, evaluating protein complexes association, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                               "Please review the evaluation of protein complexes associated with active chromatin and provide limitations.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining evaluation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Determine which protein complex is least likely to be recovered from active chromatin in the shmoo using Debate
    debate_instruction_4 = (
        "Sub-task 4: Based on the evaluation from Sub-task 3, determine which protein complex among the pre-initiation complex, pre-replication complex, enhancer protein complex, and nucleosome histone complex "
        "is least likely to be recovered from active chromatin in the shmoo of Saccharomyces cerevisiae after treatment with the quorum-sensing peptide, based on their functional roles and chromatin association patterns."
    )
    debate_roles = ["Pro Complex A", "Pro Complex B"]
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating least likely complex, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1],
                                                     "Sub-task 4: Make final decision on which protein complex is least likely to be recovered from active chromatin in the shmoo.",
                                                     is_sub_task=True)
    agents.append(f"Final Decision agent, deciding least likely complex, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
