async def forward_12(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Analyze starting material and reagents
    # Sub-task 1: Analyze (R)-(+)-Limonene structure, stereochemistry, and functional groups
    cot_instruction_1 = (
        "Sub-task 1: Analyze the structure, stereochemistry, and functional groups of (R)-(+)-Limonene, "
        "establishing the baseline molecular framework before any reaction."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing (R)-(+)-Limonene structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify and classify reagents and their typical transformations
    cot_instruction_2 = (
        "Sub-task 2: Identify and classify the reagents and conditions used in each step: Pd/C hydrogenation, m-CPBA epoxidation, sodium methoxide treatment, "
        "and propanoic acid/DCC/DMAP esterification, including their typical chemical transformations on relevant functional groups."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, classifying reagents and transformations, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Assess impact of each reaction step
    # Sub-task 3: Assess partial hydrogenation impact on double bonds and stereochemistry
    cot_instruction_3 = (
        "Sub-task 3: Assess the impact of partial hydrogenation (1 equivalent H2 consumed) of (R)-(+)-Limonene with Pd/C on the double bonds and stereochemistry, "
        "to predict the structure of product 1."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, assessing partial hydrogenation impact, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Assess epoxidation effect of m-CPBA on product 1
    cot_instruction_4 = (
        "Sub-task 4: Assess the effect of treating product 1 with 3-chloroperbenzoic acid (m-CPBA) to form product 2, "
        "focusing on epoxidation of double bonds and stereochemical outcomes."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, assessing epoxidation effect, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Assess sodium methoxide treatment on product 2
    cot_instruction_5 = (
        "Sub-task 5: Assess the effect of sodium methoxide treatment on product 2, including ring-opening of epoxides or other nucleophilic substitutions, "
        "and predict stereochemical and functional group changes leading to product 3."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4, thinking2, answer2], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, assessing sodium methoxide effect, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Assess esterification impact on product 3
    cot_instruction_6 = (
        "Sub-task 6: Assess the impact of esterification conditions (propanoic acid, DCC, catalytic DMAP) on product 3, "
        "focusing on which functional group is esterified and the resulting structure of product 4."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5, thinking2, answer2], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, assessing esterification impact, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Stage 2: Derive detailed chemical structures
    # Sub-task 7: Derive structure of product 1 after partial hydrogenation
    cot_instruction_7 = (
        "Sub-task 7: Derive the detailed chemical structure of product 1 after partial hydrogenation of (R)-(+)-Limonene, "
        "incorporating stereochemical changes and partial saturation of double bonds."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking3, answer3], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, deriving product 1 structure, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Derive structure of product 2 by epoxidation of product 1
    cot_instruction_8 = (
        "Sub-task 8: Derive the chemical structure of product 2 by applying epoxidation to product 1, "
        "including stereochemical configuration of the epoxide ring."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking4, answer4, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, deriving product 2 structure, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    # Sub-task 9: Derive structure of product 3 by sodium methoxide ring-opening
    cot_instruction_9 = (
        "Sub-task 9: Derive the chemical structure of product 3 by applying nucleophilic ring-opening or substitution reactions of sodium methoxide on product 2, "
        "with stereochemical considerations."
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking5, answer5, thinking8, answer8], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, deriving product 3 structure, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Step 9: ", sub_tasks[-1])

    # Sub-task 10: Derive structure of product 4 by esterification of product 3
    cot_instruction_10 = (
        "Sub-task 10: Derive the chemical structure of product 4 by applying esterification of product 3 with propanoic acid, DCC, and DMAP, "
        "identifying the ester linkage and stereochemistry."
    )
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking10, answer10 = await cot_agent_10([taskInfo, thinking6, answer6, thinking9, answer9], cot_instruction_10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_10.id}, deriving product 4 structure, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    print("Step 10: ", sub_tasks[-1])

    # Stage 3: Compare derived product 4 structure with multiple-choice options
    debate_instruction_11 = (
        "Sub-task 11: Compare the derived structure(s) of product 4 with the given multiple-choice options, "
        "analyzing stereochemistry, substituents, and functional groups to identify the valid structure matching the reaction sequence and conditions."
    )
    debate_agents_11 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_11 = self.max_round
    all_thinking11 = [[] for _ in range(N_max_11)]
    all_answer11 = [[] for _ in range(N_max_11)]

    for r in range(N_max_11):
        for i, agent in enumerate(debate_agents_11):
            if r == 0:
                thinking11, answer11 = await agent([taskInfo, thinking10, answer10], debate_instruction_11, r, is_sub_task=True)
            else:
                input_infos_11 = [taskInfo, thinking10, answer10] + all_thinking11[r-1] + all_answer11[r-1]
                thinking11, answer11 = await agent(input_infos_11, debate_instruction_11, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing product 4 structure with options, thinking: {thinking11.content}; answer: {answer11.content}")
            all_thinking11[r].append(thinking11)
            all_answer11[r].append(answer11)

    final_decision_agent_11 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking11, answer11 = await final_decision_agent_11([taskInfo] + all_thinking11[-1] + all_answer11[-1], "Sub-task 11: Make final decision on the valid structure of product 4.", is_sub_task=True)
    agents.append(f"Final Decision agent on product 4 structure, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    print("Step 11: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking11, answer11, sub_tasks, agents)
    return final_answer
