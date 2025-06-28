async def forward_32(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Analyze and Classify Elements

    # Sub-task 1: Analyze and identify the chemical structures and key functional groups of the reactants
    cot_instruction_1 = (
        "Sub-task 1: Analyze the chemical structures and key functional groups of the reactants: "
        "2,5-dimethylthiophene and furan-2,5-dione, including their reactive sites relevant to a [4+2] cycloaddition reaction."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                               model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing reactants, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Classify the type of [4+2] cycloaddition reaction involved
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the analysis of reactants from Sub-task 1, classify the type of [4+2] cycloaddition reaction involved, "
        "considering the nature of the diene and dienophile, and the effect of heat as a reaction condition."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                  model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, classifying [4+2] cycloaddition, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    counter_2 = Counter(possible_answers_2)
    answer2_most_common = counter_2.most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_most_common]
    answer2_final = answermapping_2[answer2_most_common]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Predict and Compare Products

    # Sub-task 3: Predict the possible cycloadduct structures formed focusing on stereochemistry and regiochemistry of the EXO product
    cot_reflect_instruction_3 = (
        "Sub-task 3: Based on the classification from Sub-task 2, predict the possible cycloadduct structures formed from the [4+2] cycloaddition between "
        "2,5-dimethylthiophene and furan-2,5-dione under heat, focusing on stereochemistry and regiochemistry of the EXO product."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                               model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                                  model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2_final, answer2_final]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, predicting EXO product structures, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback3, correct3 = await critic_agent_3([taskInfo, thinking3, answer3], 
                                                  "Please review the predicted EXO product structures for stereochemistry and regiochemistry correctness and provide limitations.", 
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining EXO product prediction, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Compare the predicted EXO product structures with the given choices
    debate_instruction_4 = (
        "Sub-task 4: Compare the predicted EXO product structures with the given choices by analyzing their stereochemical descriptors and functional groups "
        "to identify the correct product."
    )
    debate_roles = ["Proponent", "Opponent"]
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                     model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
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
            agents.append(f"Debate agent {agent.id}, round {r}, comparing predicted EXO product with choices, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                          model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], 
                                                     "Sub-task 4: Make final decision on the correct EXO product from the given choices.", 
                                                     is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct EXO product, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
