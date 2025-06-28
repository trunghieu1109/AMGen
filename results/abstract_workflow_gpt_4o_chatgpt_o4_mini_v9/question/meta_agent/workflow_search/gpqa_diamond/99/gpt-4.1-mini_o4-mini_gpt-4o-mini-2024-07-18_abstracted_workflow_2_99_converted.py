async def forward_99(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and determine structures and reactions stepwise

    # Sub-task 1: Identify and determine the structure and nature of Compound A (C3H6)
    cot_instruction_1 = (
        "Sub-task 1: Identify and determine the structure and nature of Compound A (C3H6) based on its molecular formula and typical isomers, "
        "to establish the starting material for the reaction sequence."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying Compound A, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze bromination of Compound A to form Compound B
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on Compound A structure, analyze the bromination reaction in presence of carbon tetrachloride to determine the structure and properties of Compound B, "
        "including type of addition and stereochemistry if relevant."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing bromination, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose most consistent answer
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Determine reaction of Compound B with alcoholic KOH to form Compound C
    cot_instruction_3 = (
        "Sub-task 3: Determine the reaction of Compound B with alcoholic KOH to form Compound C, identifying the reaction type (e.g., elimination) and the structure and physical state (gas, liquid) of Compound C."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, determining reaction with alcoholic KOH, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Analyze thermal decomposition of Compound C through red-hot iron tube to form Compound D
    debate_instruction_4 = (
        "Sub-task 4: Analyze the thermal decomposition of Compound C by passing it through a red-hot iron tube to form Compound D, "
        "identifying the product structure and relevant spectroscopic features such as 1H NMR signals."
    )
    debate_roles_4 = ["Proposer", "Opponent"]
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles_4]
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
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing thermal decomposition, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on Compound D structure and NMR features.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding Compound D structure and NMR, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Reaction of Compound D with mixture of two strong acids to form Compound E
    cot_instruction_5 = (
        "Sub-task 5: Examine the reaction of Compound D with a mixture of two strong acids to form Compound E, "
        "identifying the nature of the acids, the reaction mechanism, and the structure of Compound E."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, analyzing reaction with strong acids, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Reaction of Compound E with iron scrap and HCl to form Compound F
    cot_instruction_6 = (
        "Sub-task 6: Determine the reaction of Compound E with iron scrap and hydrochloric acid to form Compound F, "
        "identifying the product structure and its common applications, such as in dye synthesis."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, determining reaction with iron scrap and HCl, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Reaction of Compound F with nitrous acid to form Compound G
    cot_instruction_7 = (
        "Sub-task 7: Analyze the reaction of Compound F with nitrous acid to form Compound G, "
        "identifying the functional group transformations and the structure of Compound G."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, analyzing reaction with nitrous acid, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Reaction of Compound G with sodium hydroxide to form Compound H
    cot_instruction_8 = (
        "Sub-task 8: Determine the reaction of Compound G with sodium hydroxide to form Compound H, "
        "identifying the structure of Compound H and its chemical properties, including color reactions with ferric chloride solution."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, determining reaction with NaOH and color reaction, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    # Stage 2: Evaluate spectroscopic data, applications, chemical behavior, and physical properties

    # Sub-task 9: Evaluate if Compound D gives two singlets in 1H NMR
    cot_instruction_9 = (
        "Sub-task 9: Evaluate the spectroscopic data and chemical properties of Compound D, specifically whether it gives two singlets in the 1H NMR spectrum, "
        "based on its structure determined in previous steps."
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking4, answer4], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, evaluating NMR of Compound D, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Step 9: ", sub_tasks[-1])

    # Sub-task 10: Assess practical applications of Compound F in dye synthesis
    cot_instruction_10 = (
        "Sub-task 10: Assess the practical applications of Compound F, particularly its use in dye synthesis, "
        "based on its structure and chemical nature established earlier."
    )
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking10, answer10 = await cot_agent_10([taskInfo, thinking6, answer6], cot_instruction_10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_10.id}, assessing dye synthesis use of Compound F, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    print("Step 10: ", sub_tasks[-1])

    # Sub-task 11: Evaluate if Compound H gives yellow color with ferric chloride
    cot_instruction_11 = (
        "Sub-task 11: Evaluate the chemical behavior of Compound H with ferric chloride solution to verify if it produces a yellow color, "
        "based on its functional groups and structure."
    )
    cot_agent_11 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking11, answer11 = await cot_agent_11([taskInfo, thinking8, answer8], cot_instruction_11, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_11.id}, evaluating color reaction of Compound H, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    print("Step 11: ", sub_tasks[-1])

    # Sub-task 12: Confirm physical state and flammability of Compound C
    cot_instruction_12 = (
        "Sub-task 12: Confirm the physical state and flammability of Compound C, specifically whether it is a flammable gas, "
        "based on its structure and known chemical properties."
    )
    cot_agent_12 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking12, answer12 = await cot_agent_12([taskInfo, thinking3, answer3], cot_instruction_12, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_12.id}, confirming physical state and flammability of Compound C, thinking: {thinking12.content}; answer: {answer12.content}")
    sub_tasks.append(f"Sub-task 12 output: thinking - {thinking12.content}; answer - {answer12.content}")
    print("Step 12: ", sub_tasks[-1])

    # Sub-task 13: Compare all evaluated statements about D, F, H, and C to identify incorrect statement
    cot_reflect_instruction_13 = (
        "Sub-task 13: Compare all evaluated statements about compounds D, F, H, and C to identify which statement is incorrect, "
        "integrating all previous findings from Sub-tasks 9, 10, 11, and 12."
    )
    cot_agent_13 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_13 = [taskInfo, thinking9, answer9, thinking10, answer10, thinking11, answer11, thinking12, answer12]
    thinking13, answer13 = await cot_agent_13(cot_inputs_13, cot_reflect_instruction_13, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_13.id}, integrating all evaluations to identify incorrect statement, thinking: {thinking13.content}; answer: {answer13.content}")
    sub_tasks.append(f"Sub-task 13 output: thinking - {thinking13.content}; answer - {answer13.content}")
    print("Step 13: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking13, answer13, sub_tasks, agents)
    return final_answer
