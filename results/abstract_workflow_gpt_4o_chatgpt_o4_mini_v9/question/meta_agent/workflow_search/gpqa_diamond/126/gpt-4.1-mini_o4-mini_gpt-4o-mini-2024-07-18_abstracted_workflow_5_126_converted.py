async def forward_126(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Extract and characterize structural features of starting and choice molecules
    # Sub-task 1: Extract features of starting molecule using Chain-of-Thought
    cot_instruction_1 = (
        "Sub-task 1: Extract and characterize the structural features of the starting molecule 5-butylnona-2,6-diene, "
        "including chain length, substituents, and positions of double bonds, to understand its chemical nature before heating."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting features of starting molecule, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 0.1: ", sub_tasks[-1])

    # Sub-task 2: Extract features of choice molecules using Chain-of-Thought
    cot_instruction_2 = (
        "Sub-task 2: Extract and characterize the structural features of each choice molecule (4-ethyl-3-methyldeca-1,5-diene, 5-ethylundeca-2,6-diene, 5-ethyl-4-methyldeca-2,6-diene), "
        "including chain length, substituents, and double bond positions to enable comparison with the starting molecule."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, extracting features of choice molecules, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Analyze possible chemical transformations and compare with choices
    # Sub-task 3: Analyze heating transformations using Self-Consistency Chain-of-Thought
    cot_sc_instruction_3 = (
        "Sub-task 3: Analyze the possible chemical transformations that occur when 5-butylnona-2,6-diene is heated, "
        "focusing on typical reactions of dienes such as isomerization, rearrangement, or polymerization, to predict plausible product structures."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, analyzing heating transformations, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    most_common_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[most_common_answer_3]
    answer3 = answermapping_3[most_common_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 1.3: ", sub_tasks[-1])

    # Sub-task 4: Compare predicted products with choice molecules using Chain-of-Thought
    cot_instruction_4 = (
        "Sub-task 4: Classify and compare the structural features of the predicted product(s) from heating 5-butylnona-2,6-diene "
        "with the features of the choice molecules to identify which choice(s) could correspond to the product formed."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, comparing predicted products with choices, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Generate product variants and compare with choices
    # Sub-task 5: Generate product variants applying heating transformations using Debate
    debate_instruction_5 = (
        "Sub-task 5: Apply the identified heating-induced transformations to the starting molecule's structure to generate possible product variants, "
        "including changes in chain length, substituent positions, and double bond locations."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3, answer3], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answer3] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, generating product variants, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on generated product variants.", is_sub_task=True)
    agents.append(f"Final Decision agent on product variants, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 2.5: ", sub_tasks[-1])

    # Sub-task 6: Compare generated variants with choice molecules using Chain-of-Thought
    cot_instruction_6 = (
        "Sub-task 6: Compare the generated product variants with the choice molecules to determine structural matches or close analogs, "
        "focusing on substituent identity and position, chain length, and double bond placement."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking2, answer2, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, comparing generated variants with choices, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 2.6: ", sub_tasks[-1])

    # Stage 3: Evaluate and prioritize choice molecules
    # Sub-task 7: Evaluate and prioritize using Debate
    debate_instruction_7 = (
        "Sub-task 7: Evaluate and prioritize the choice molecules based on their structural compatibility with the predicted product(s) formed by heating 5-butylnona-2,6-diene, "
        "selecting the most likely product formed under heating conditions."
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
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating and prioritizing choices, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the most likely product formed.", is_sub_task=True)
    agents.append(f"Final Decision agent on prioritizing choices, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 3.7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
