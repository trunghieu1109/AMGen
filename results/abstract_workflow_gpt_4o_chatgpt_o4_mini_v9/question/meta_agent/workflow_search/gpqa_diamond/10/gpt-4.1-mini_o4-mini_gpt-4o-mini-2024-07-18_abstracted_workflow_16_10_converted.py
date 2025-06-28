async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    cot_sc_instruction_1 = (
        "Sub-task 1: Analyze the molecular biology concepts related to programmed ribosomal frameshifting in SARS-CoV-2, "
        "including the mechanism involving slippery nucleotides and pseudoknot structures, and compare the conformation of SARS-CoV-2 frameshifting signals with those of SARS-CoV."
    )
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_sc_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, analyzing programmed ribosomal frameshifting, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1.content)
        thinkingmapping_1[answer1.content] = thinking1
        answermapping_1[answer1.content] = answer1
    counter_1 = Counter(possible_answers_1)
    answer1_final = counter_1.most_common(1)[0][0]
    thinking1_final = thinkingmapping_1[answer1_final]
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_final.content}; answer - {answer1_final}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    cot_sc_instruction_2 = (
        "Sub-task 2: Examine the role and molecular function of the SARS-CoV-2 nsp10/nsp14 complex, specifically focusing on the exonuclease (ExoN) activity, heterodimer formation, "
        "and its involvement in mismatch repair and dsRNA stability."
    )
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_sc_agents_2[i]([taskInfo], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, examining nsp10/nsp14 complex, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    counter_2 = Counter(possible_answers_2)
    answer2_final = counter_2.most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_final]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    cot_sc_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    cot_sc_instruction_3 = (
        "Sub-task 3: Investigate the apoptotic pathways triggered by SARS-CoV-2 ORF3a protein, focusing on caspase-8 activation, the extrinsic apoptotic pathway, "
        "and the role of Bcl-2 in the mitochondrial (intrinsic) apoptotic pathway to understand the mechanism of apoptosis induction."
    )
    for i in range(self.max_sc):
        thinking3, answer3 = await cot_sc_agents_3[i]([taskInfo], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3[i].id}, investigating ORF3a apoptotic pathways, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    counter_3 = Counter(possible_answers_3)
    answer3_final = counter_3.most_common(1)[0][0]
    thinking3_final = thinkingmapping_3[answer3_final]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    cot_sc_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    cot_sc_instruction_4 = (
        "Sub-task 4: Evaluate the relationship between the rate of programmed -1 ribosomal frameshifting in vitro and the number of conformations adopted by the pseudoknot structures in SARS-CoV and SARS-CoV-2, "
        "including the presence of multiple conformations under tension and their correlation with frameshifting efficiency."
    )
    for i in range(self.max_sc):
        thinking4, answer4 = await cot_sc_agents_4[i]([taskInfo], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_4[i].id}, evaluating frameshifting rate and pseudoknot conformations, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    counter_4 = Counter(possible_answers_4)
    answer4_final = counter_4.most_common(1)[0][0]
    thinking4_final = thinkingmapping_4[answer4_final]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_final.content}; answer - {answer4_final}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: Compare and contrast the findings from subtasks 1 to 4 to identify inconsistencies or inaccuracies in the provided statements about SARS-CoV-2 molecular biology, "
        "with the goal of determining which statement is incorrect."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent(
                    [taskInfo, thinking1_final, answer1_final, thinking2_final, answer2_final, thinking3_final, answer3_final, thinking4_final, answer4_final],
                    debate_instruction_5, r, is_sub_task=True
                )
            else:
                input_infos_5 = [taskInfo, thinking1_final, answer1_final, thinking2_final, answer2_final, thinking3_final, answer3_final, thinking4_final, answer4_final] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, identifying incorrect statement, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5(
        [taskInfo] + all_thinking5[-1] + all_answer5[-1],
        "Sub-task 5: Make final decision on which statement about SARS-CoV-2 molecular biology is incorrect.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, determining incorrect statement, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
