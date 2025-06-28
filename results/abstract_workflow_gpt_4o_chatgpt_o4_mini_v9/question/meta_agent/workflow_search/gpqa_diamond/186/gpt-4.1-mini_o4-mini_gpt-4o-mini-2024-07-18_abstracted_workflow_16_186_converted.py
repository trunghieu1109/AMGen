async def forward_186(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Gather and summarize instrument parameters and star data
    # Sub-task 1: Gather ESPRESSO instrument parameters and observational conditions
    cot_instruction_1 = (
        "Sub-task 1: Gather and summarize all relevant instrument parameters and observational conditions "
        "for the ESPRESSO spectrograph coupled with an 8m VLT telescope at Paranal Observatory, including sensitivity, "
        "wavelength coverage, pixel binning, and exposure time requirements to achieve S/N ≥ 10 per binned pixel in 1 hour."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, gathered instrument parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Collect astrophysical data for each star (apparent magnitudes or compute from absolute magnitude and distance, coordinates)
    cot_instruction_2 = (
        "Sub-task 2: Collect and compile the astrophysical data for each star listed: apparent magnitudes (or compute from absolute magnitude and distance), "
        "coordinates (RA, DEC), and any other relevant parameters needed to estimate their brightness as seen from Paranal Observatory."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, collected star data, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Calculate apparent V magnitude for stars with given absolute magnitude and distance
    cot_sc_instruction_3 = (
        "Sub-task 3: Calculate the apparent V magnitude for each star with given absolute magnitude and distance, "
        "using the distance modulus formula, to determine their brightness as observed from Earth."
    )
    N_sc = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, calculating apparent magnitudes, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose the most consistent answer (majority vote)
    from collections import Counter
    answer_counts_3 = Counter(possible_answers_3)
    best_answer_3 = answer_counts_3.most_common(1)[0][0]
    thinking3 = thinkingmapping_3[best_answer_3]
    answer3 = answermapping_3[best_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Evaluate visibility and observability constraints of each star from Paranal Observatory
    cot_instruction_4 = (
        "Sub-task 4: Evaluate the visibility and observability constraints of each star from Paranal Observatory, "
        "considering their RA and DEC, to confirm if they can be observed during a typical observing night with ESPRESSO on the VLT."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluated visibility, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Estimate S/N and determine detectability
    # Sub-task 5: Estimate expected S/N per binned pixel for each star during 1-hour exposure
    cot_instruction_5 = (
        "Sub-task 5: Estimate the expected signal-to-noise ratio (S/N) per binned pixel for each star during a 1-hour exposure with ESPRESSO on the 8m VLT, "
        "using the instrument sensitivity parameters and the stars apparent magnitude and visibility conditions."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking1, answer1, thinking3, answer3, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, estimated S/N, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Determine which stars meet detectability criterion S/N ≥ 10
    cot_instruction_6 = (
        "Sub-task 6: Determine which stars meet the detectability criterion of S/N ≥ 10 per binned pixel in 1 hour exposure, "
        "based on the S/N estimates from Sub-task 5."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, determined detectable stars, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Count total detectable stars and compare with answer choices
    debate_instruction_7 = (
        "Sub-task 7: Count the total number of stars from the list that are detectable according to the criterion and compare this count with the provided answer choices (2, 3, 4, 5). "
        "Provide reasoning for the final choice."
    )
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]

    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, counting detectable stars and comparing choices, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)

    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the number of detectable stars.", is_sub_task=True)
    agents.append(f"Final Decision agent on counting detectable stars, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
