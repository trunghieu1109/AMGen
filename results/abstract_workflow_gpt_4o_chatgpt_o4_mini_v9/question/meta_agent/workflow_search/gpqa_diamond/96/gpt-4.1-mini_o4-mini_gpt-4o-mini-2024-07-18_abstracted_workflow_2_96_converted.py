async def forward_96(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements
    # Sub-task 1: Analyze biological context of meiosis and embryogenesis
    cot_instruction_1 = (
        "Sub-task 1: Analyze the biological context of the question, specifically the processes of meiosis and embryogenesis, "
        "and understand the difference between meiosis and mitosis in terms of chromosome separation and haploid gamete formation. "
        "This provides foundational knowledge about the origin of chromosomal abnormalities like Klinefelters and Downs syndromes."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing meiosis and embryogenesis, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Identify and characterize chromosomal abnormalities in Klinefelters and Downs syndromes
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, identify and characterize the chromosomal abnormalities underlying "
        "Klinefelters syndrome and Downs syndrome, focusing on their karyotypic differences and how these abnormalities arise during meiosis or fertilization."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, characterizing chromosomal abnormalities, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Select the most consistent answer by frequency
    answer2_counter = Counter(possible_answers_2)
    answer2_final = answer2_counter.most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_final]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Analyze phenotypic consequences of Klinefelters vs Downs syndrome
    cot_instruction_3 = (
        "Sub-task 3: Analyze the phenotypic consequences of Klinefelters syndrome compared to Downs syndrome, "
        "emphasizing why Klinefelters syndrome has less prominent phenotypic effects despite being a chromosomal abnormality."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2_final, answer2_final], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing phenotypic consequences, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Evaluate molecular mechanisms and select responsible one
    # Sub-task 4: Evaluate molecular mechanisms in context of meiosis, embryogenesis, epigenetics
    cot_instruction_4 = (
        "Sub-task 4: Evaluate the molecular mechanisms listed in the answer choices (polymerase alpha progression, spindle attachment to kinetochores, "
        "chromatin methylation by histone methyltransferases, chiasmata resolution by separase) in the context of their role in meiosis, embryogenesis, and epigenetic regulation, "
        "to determine which mechanism could explain the less severe phenotype of Klinefelters syndrome."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating molecular mechanisms, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Select molecular mechanism responsible for less prominent phenotype of Klinefelters
    debate_instruction_5 = (
        "Sub-task 5: Based on the output of Sub-task 4, debate and select the molecular mechanism responsible for the less prominent phenotypic consequences "
        "of Klinefelters syndrome compared to Downs syndrome, considering biological relevance to chromosomal abnormalities and gene expression regulation."
    )
    debate_roles = ["Proponent", "Opponent"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating molecular mechanism, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the molecular mechanism responsible for less prominent phenotype of Klinefelters syndrome.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final molecular mechanism, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
