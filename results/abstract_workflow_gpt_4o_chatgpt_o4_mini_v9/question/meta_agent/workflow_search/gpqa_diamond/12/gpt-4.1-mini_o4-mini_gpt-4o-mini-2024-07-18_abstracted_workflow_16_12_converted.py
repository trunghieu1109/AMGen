async def forward_12(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Sequential Analysis of Reaction Steps with Chain-of-Thought

    # Sub-task 1: Analyze (R)-(+)-Limonene stereochemistry and structure
    cot_instruction_1 = (
        "Sub-task 1: Analyze the starting material (R)-(+)-Limonene, determine its stereochemistry and structure "
        "to establish the baseline molecule before any reactions."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzed (R)-(+)-Limonene, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Hydrogenation of (R)-(+)-Limonene with Pd/C and H2 (1 equiv), deduce product 1 structure and stereochemistry
    cot_instruction_2 = (
        "Sub-task 2: Based on Sub-task 1 output, analyze the hydrogenation of (R)-(+)-Limonene with Pd/C under hydrogen atmosphere "
        "until 1 equivalent of hydrogen is consumed, deduce the structure and stereochemistry of product 1."
    )
    N_sc = self.max_sc if hasattr(self, 'max_sc') else 5
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzed hydrogenation, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Reaction of product 1 with m-CPBA to form product 2, identify oxidation type and stereochemistry
    cot_instruction_3 = (
        "Sub-task 3: Based on product 1 structure, analyze the reaction with 3-chloroperbenzoic acid (m-CPBA) to form product 2, "
        "identify the type of oxidation, resulting functional groups, and stereochemistry."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzed m-CPBA oxidation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Reaction of product 2 with sodium methoxide to form product 3, determine chemical transformation
    cot_instruction_4 = (
        "Sub-task 4: Analyze the reaction of product 2 with sodium methoxide to form product 3, "
        "determine the chemical transformation, changes in functional groups, and stereochemistry."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, analyzed sodium methoxide reaction, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Reaction of product 3 with propanoic acid, DCC, and catalytic DMAP to form product 4, identify reaction type and stereochemistry
    cot_instruction_5 = (
        "Sub-task 5: Analyze the reaction of product 3 with propanoic acid, dicyclohexylcarbodiimide (DCC), and catalytic 4-dimethylaminopyridine (DMAP) to form product 4, "
        "identify the type of reaction (esterification or acylation), resulting functional groups, and stereochemistry."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, analyzed esterification/acylation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Integration and Final Selection with Reflexion and Debate

    # Sub-task 6: Integrate all structural and stereochemical info from products 1-4 to deduce valid structure(s) of product 4
    cot_reflect_instruction_6 = (
        "Sub-task 6: Integrate all structural and stereochemical information from products 1 through 4 to deduce the valid structure(s) of product 4, "
        "considering possible isomers and stereochemical outcomes."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round if hasattr(self, 'max_round') else 3
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, integrating structures, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback6, correct6 = await critic_agent_6([taskInfo, thinking6, answer6],
                                                  "Please review the validity of the deduced product 4 structures and provide limitations.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, feedback on integration, thinking: {feedback6.content}; answer: {correct6.content}")
        if correct6.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining integration, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Compare deduced product 4 structures with multiple-choice options and select correct answer using Debate
    debate_instruction_7 = (
        "Sub-task 7: Compare the deduced structure(s) of product 4 with the provided multiple-choice options, "
        "evaluate stereochemistry, substituents, and functional groups to select the correct answer."
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
            agents.append(f"Debate agent {agent.id}, round {r}, comparing options, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the correct product 4 structure.", is_sub_task=True)
    agents.append(f"Final Decision agent on product 4 selection, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
