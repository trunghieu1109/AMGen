async def forward_117(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements

    # Sub-task 1: Analyze chemical structure and functional groups of 4,4-dimethylcyclopent-1-enol
    cot_instruction_1 = (
        "Sub-task 1: Analyze the chemical structure and functional groups of 4,4-dimethylcyclopent-1-enol to understand its reactive sites and properties relevant to bromine reaction."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing 4,4-dimethylcyclopent-1-enol, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze chemical nature and reactivity of bromine with enols and alkenes
    cot_instruction_2 = (
        "Sub-task 2: Analyze the chemical nature and reactivity of bromine, especially in reactions with enols and alkenes, to identify possible reaction pathways."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing bromine reactivity, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Classify types of reactions between 4,4-dimethylcyclopent-1-enol and bromine
    cot_instruction_3 = (
        "Sub-task 3: Classify the types of reactions that can occur between an enol (4,4-dimethylcyclopent-1-enol) and bromine, including electrophilic addition, substitution, and oxidation, based on outputs from Sub-task 1 and 2."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Self-Consistency Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    N = self.max_sc
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agent_3.id}, classifying reaction types, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    most_common_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[most_common_answer_3]
    answer3 = answermapping_3[most_common_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Evaluate, Predict, and Compare Products

    # Sub-task 4: Evaluate possible reaction mechanisms based on functional groups and bromine reactivity
    cot_instruction_4 = (
        "Sub-task 4: Evaluate the possible reaction mechanisms between 4,4-dimethylcyclopent-1-enol and bromine based on the functional groups and bromine reactivity, focusing on which mechanism is most likely to dominate, using output from Sub-task 3."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating reaction mechanisms, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Predict major product(s) formed applying the most probable mechanism
    cot_instruction_5 = (
        "Sub-task 5: Predict the major product(s) formed from the reaction between 4,4-dimethylcyclopent-1-enol and bromine by applying the most probable reaction mechanism identified in Sub-task 4."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, predicting major product, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compare predicted product(s) with given choices to select correct answer
    debate_instruction_6 = (
        "Sub-task 6: Compare the predicted major product(s) with the given choices (2-bromo-4,4-dimethylcyclopentanone, (1R,2R)-1,2-dibromo-4,4-dimethylcyclopentanol, (1R,2S)-1,2-dibromo-4,4-dimethylcyclopentanol, 4-bromo-4,4-dimethylcyclentanone) to select the correct answer."
    )
    debate_roles = ["Proponent", "Opponent"]
    debate_agents_6 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in debate_roles
    ]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating product selection, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the correct major product from given choices.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct product, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
