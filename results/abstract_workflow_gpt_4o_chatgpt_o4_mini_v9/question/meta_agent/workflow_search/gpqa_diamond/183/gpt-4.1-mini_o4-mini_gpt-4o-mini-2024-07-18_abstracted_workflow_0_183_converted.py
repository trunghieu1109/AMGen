async def forward_183(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Analyze molecule, reagents, and directing effects
    # Sub-task 1: Analyze target molecule functional groups and substitution pattern (CoT)
    cot_instruction_1 = (
        "Sub-task 1: Analyze the target molecule 2-(tert-butyl)-1-ethoxy-3-nitrobenzene to identify its functional groups, "
        "substitution pattern on the benzene ring, and the nature of each substituent (tert-butyl, ethoxy, nitro) with their relative positions."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing target molecule, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze reagents and reaction conditions in all four choices (Self-Consistency CoT)
    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze each reagent and reaction condition listed in all four choices to classify their chemical role "
        "(e.g., electrophilic aromatic substitution, reduction, diazotization, nucleophilic substitution, sulfonation, desulfonation) "
        "and expected effect on benzene or substituted benzene derivatives."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing reagents and conditions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer by frequency
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Identify directing effects of substituents (tert-butyl, ethoxy) (Reflexion)
    cot_reflect_instruction_3 = (
        "Sub-task 3: Based on outputs from Sub-task 1 and Sub-task 2, identify the directing effects (ortho/para or meta) "
        "of substituents introduced at each step, especially tert-butyl and ethoxy groups, to predict regioselectivity of subsequent electrophilic aromatic substitution reactions."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, identifying directing effects, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback3, correct3 = await critic_agent_3([taskInfo, thinking3, answer3],
                                                  "Please review the directing effects identification and provide limitations.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining directing effects, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 1: Assess impact of reaction steps
    # Sub-task 4: Impact of tert-butyl group introduction (CoT)
    cot_instruction_4 = (
        "Sub-task 4: Assess the impact of introducing the tert-butyl group first on benzene using tert-butyl chloride/AlCl3, "
        "considering its activating and ortho/para directing effects for subsequent substitutions."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, assessing tert-butyl introduction, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Effect of nitration on tert-butylbenzene (Debate)
    debate_instruction_5 = (
        "Sub-task 5: Evaluate the effect of nitration (HNO3/H2SO4) on the tert-butylbenzene intermediate, "
        "predicting the position of nitro group introduction based on directing effects of tert-butyl substituent."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            input_infos_5 = [taskInfo, thinking4, answer4]
            if r > 0:
                input_infos_5 += all_thinking5[r-1] + all_answer5[r-1]
            thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating nitration effect, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1],
                                                    "Sub-task 5: Make final decision on nitration position and effect.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, nitration effect decision, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Effect of reduction (Fe/HCl) on nitro group (CoT)
    cot_instruction_6 = (
        "Sub-task 6: Evaluate the effect of reduction (Fe/HCl) on the nitro group to an amino group, "
        "and how this transformation affects further substitution steps."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, evaluating reduction effect, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Diazotization and ethoxy introduction (Reflexion)
    cot_reflect_instruction_7 = (
        "Sub-task 7: Assess the diazotization step (NaNO2/HCl) on the amino group and subsequent transformations "
        "(e.g., hydrolysis, substitution) to introduce the ethoxy group via nucleophilic substitution (NaOH/EtBr)."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking6, answer6]
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, assessing diazotization and ethoxy introduction, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        feedback7, correct7 = await critic_agent_7([taskInfo, thinking7, answer7],
                                                  "Review the diazotization and ethoxy introduction step and provide limitations.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback7])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining diazotization step, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Role and timing of sulfonation and desulfonation (CoT)
    cot_instruction_8 = (
        "Sub-task 8: Evaluate the role and timing of sulfonation (SO3/H2SO4) and desulfonation (dilute H2SO4) steps in the sequence, "
        "and their impact on directing groups and substitution pattern."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking5, answer5, thinking6, answer6], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, evaluating sulfonation/desulfonation, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    # Stage 2: Derive correct sequence and compare with choices
    # Sub-task 9: Derive correct sequence of reactions (Reflexion)
    cot_reflect_instruction_9 = (
        "Sub-task 9: Derive the correct sequence of reactions that leads to the high-yield synthesis of 2-(tert-butyl)-1-ethoxy-3-nitrobenzene starting from benzene, "
        "by applying the knowledge of directing effects, reaction compatibility, and functional group transformations from previous subtasks."
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_9 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_9 = self.max_round
    cot_inputs_9 = [taskInfo, thinking4, answer4, thinking5, answer5, thinking6, answer6, thinking7, answer7, thinking8, answer8]
    thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_9.id}, deriving correct sequence, thinking: {thinking9.content}; answer: {answer9.content}")
    for i in range(N_max_9):
        feedback9, correct9 = await critic_agent_9([taskInfo, thinking9, answer9],
                                                  "Review the derived reaction sequence and provide limitations.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_9.id}, providing feedback, thinking: {feedback9.content}; answer: {correct9.content}")
        if correct9.content == "True":
            break
        cot_inputs_9.extend([thinking9, answer9, feedback9])
        thinking9, answer9 = await cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_9.id}, refining sequence, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Step 9: ", sub_tasks[-1])

    # Sub-task 10: Compare derived sequence with given choices (Debate)
    debate_instruction_10 = (
        "Sub-task 10: Compare the derived correct sequence with each of the four given choices to identify which option matches the optimal reaction order and conditions "
        "for high yield synthesis of the target molecule."
    )
    debate_agents_10 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_10 = self.max_round
    all_thinking10 = [[] for _ in range(N_max_10)]
    all_answer10 = [[] for _ in range(N_max_10)]
    for r in range(N_max_10):
        for i, agent in enumerate(debate_agents_10):
            input_infos_10 = [taskInfo, thinking9, answer9]
            if r > 0:
                input_infos_10 += all_thinking10[r-1] + all_answer10[r-1]
            thinking10, answer10 = await agent(input_infos_10, debate_instruction_10, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing sequences, thinking: {thinking10.content}; answer: {answer10.content}")
            all_thinking10[r].append(thinking10)
            all_answer10[r].append(answer10)
    final_decision_agent_10 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking10, answer10 = await final_decision_agent_10([taskInfo] + all_thinking10[-1] + all_answer10[-1],
                                                      "Sub-task 10: Make final decision on the correct choice option for the synthesis sequence.",
                                                      is_sub_task=True)
    agents.append(f"Final Decision agent, final choice decision, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    print("Step 10: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer
