async def forward_25(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements
    # Sub-task 1: Analyze the structure and defining attributes of the given dienes
    cot_instruction_1 = (
        "Sub-task 1: Analyze the structure and defining attributes of the given dienes (1. 2,3-dimethylbuta-1,3-diene, "
        "2. (2E,4E)-hexa-2,4-diene, 3. cyclopenta-1,3-diene, 4. (2Z,4Z)-hexa-2,4-diene) to understand their conjugation, substitution pattern, "
        "and stereochemistry, which influence their reactivity."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing dienes structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze the possible reactants (A) given in the options
    cot_instruction_2 = (
        "Sub-task 2: Analyze the possible reactants (A) given in the options (4,4-diiodocyclobut-2-en-1-one and 2,2-diiodoethen-1-one) "
        "to understand their structure and how they might react with cyclohexene to form the product 8,8-diiodobicyclo[4.2.0]octan-7-one."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing reactants, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Evaluate Reactivity and Reactant Compatibility
    # Sub-task 3: Evaluate the reactivity of each diene based on their structural features
    cot_sc_instruction_3 = (
        "Sub-task 3: Based on the analysis of dienes from Sub-task 1, evaluate the reactivity of each diene focusing on conjugation, ring strain, substitution, "
        "and stereochemistry, to rank them from most reactive to least reactive in the context of the given reaction."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, evaluating diene reactivity, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose the most frequent answer as consensus
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3_content = thinkingmapping_3[answer3_content].content
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_content}; answer - {answer3_content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Determine which reactant (A) is consistent with the formation of the product
    cot_sc_instruction_4 = (
        "Sub-task 4: Analyze the reaction mechanism and compatibility of each reactant (A) from Sub-task 2 with cyclohexene to form 8,8-diiodobicyclo[4.2.0]octan-7-one, "
        "and determine the correct reactant."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2, answer2], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, analyzing reactant compatibility, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    # Choose the most frequent answer as consensus
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4_content = thinkingmapping_4[answer4_content].content
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_content}; answer - {answer4_content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 3: Integrate findings and select correct choice
    # Sub-task 5: Debate to integrate diene reactivity order and correct reactant
    debate_instruction_5 = (
        "Sub-task 5: Integrate the findings from Sub-task 3 (diene reactivity order) and Sub-task 4 (correct reactant A) to select the correct choice among the given options "
        "that matches both the reactant and the reactivity sequence."
    )
    debate_roles = ["Proponent", "Opponent"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3, answermapping_3[answer3_content], thinking4, answermapping_4[answer4_content]], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3, answermapping_3[answer3_content], thinking4, answermapping_4[answer4_content]] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating reactivity and reactant, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct reactant and diene reactivity sequence.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
