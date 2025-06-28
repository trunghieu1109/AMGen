async def forward_74(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Analyze the construct sequence and identify key features

    # Sub-task 1: Analyze the plus strand DNA sequence to identify the coding sequence for GADD45G fused with HA tag at N-terminus
    cot_instruction_1 = (
        "Sub-task 1: Analyze the provided plus strand DNA sequence of the construct to identify the coding sequence for the GADD45G gene "
        "fused with the influenza hemagglutinin (HA) antigenic determinant at the N-terminus, including the start codon and any relevant sequence features."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing plus strand DNA sequence, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify the exact nucleotide sequence corresponding to the influenza hemagglutinin antigenic determinant and verify its integrity
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, identify the exact nucleotide sequence corresponding to the influenza hemagglutinin antigenic determinant within the given construct sequence "
        "and verify its integrity and correctness compared to the known HA epitope sequence."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, verifying HA antigenic determinant, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer (majority vote)
    answer2_final = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_final]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Translation and sequence evaluation

    # Sub-task 3: Translate the plus strand DNA sequence into amino acid sequence to check for premature stop codons or mutations
    cot_instruction_3 = (
        "Sub-task 3: Translate the plus strand DNA sequence into the corresponding amino acid sequence to check for any premature stop codons or mutations that could cause early termination of translation."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, translating DNA to protein, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Evaluate whether the translated protein contains the correct HA antigenic determinant sequence without missense mutations
    cot_instruction_4 = (
        "Sub-task 4: Evaluate whether the translated protein sequence contains the correct HA antigenic determinant sequence without missense mutations that could affect protein expression or function."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2_final, answer2_final, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating HA antigenic determinant in protein, thinking: {thinking4.content}; answer: {answer4.content}")

    # Sub-task 5: Assess if the construct includes a suitable linker sequence between HA tag and GADD45G coding sequence
    cot_instruction_5 = (
        "Sub-task 5: Assess if the construct sequence includes a suitable linker sequence between the HA tag and GADD45G coding sequence to prevent proteolysis of the nascent fusion protein."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking1, answer1], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, assessing linker sequence presence, thinking: {thinking5.content}; answer: {answer5.content}")

    # Sub-task 6: Check for premature stop codons and evaluate biological plausibility of tRNA availability in mouse
    cot_instruction_6 = (
        "Sub-task 6: Check for the presence of any premature stop codons (e.g., UAA) in the coding sequence and evaluate the biological plausibility of tRNA availability for such codons in the mouse model to rule out translation termination issues."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking3, answer3], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, checking premature stop codons and tRNA availability, thinking: {thinking6.content}; answer: {answer6.content}")

    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 4-6: ", sub_tasks[-3:])

    # Stage 2: Integration and final explanation

    # Sub-task 7: Integrate findings from translation, HA verification, linker presence, and stop codon evaluation
    cot_reflect_instruction_7 = (
        "Sub-task 7: Integrate findings from the translation analysis, HA sequence verification, linker presence, and stop codon evaluation to determine the most likely cause of the failure to overexpress the protein in the transfected cells."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking4, answer4, thinking5, answer5, thinking6, answer6]
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, integrating findings, thinking: {thinking7.content}; answer: {answer7.content}")

    for i in range(N_max_7):
        feedback7, correct7 = await critic_agent_7([taskInfo, thinking7, answer7],
                                                  "please review the integration of findings and provide limitations.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback7])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining integration, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Based on integrated analysis, select the correct explanation from provided choices
    debate_instruction_8 = (
        "Sub-task 8: Based on the integrated analysis, select the correct explanation from the provided choices regarding why the protein overexpression failed."
    )
    debate_agents_8 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_8 = self.max_round
    all_thinking8 = [[] for _ in range(N_max_8)]
    all_answer8 = [[] for _ in range(N_max_8)]

    for r in range(N_max_8):
        for i, agent in enumerate(debate_agents_8):
            if r == 0:
                thinking8, answer8 = await agent([taskInfo, thinking7, answer7], debate_instruction_8, r, is_sub_task=True)
            else:
                input_infos_8 = [taskInfo, thinking7, answer7] + all_thinking8[r-1] + all_answer8[r-1]
                thinking8, answer8 = await agent(input_infos_8, debate_instruction_8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting final explanation, thinking: {thinking8.content}; answer: {answer8.content}")
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)

    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + all_thinking8[-1] + all_answer8[-1],
                                                     "Sub-task 8: Make final decision on the correct explanation for protein overexpression failure.",
                                                     is_sub_task=True)
    agents.append(f"Final Decision agent, making final explanation decision, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer
