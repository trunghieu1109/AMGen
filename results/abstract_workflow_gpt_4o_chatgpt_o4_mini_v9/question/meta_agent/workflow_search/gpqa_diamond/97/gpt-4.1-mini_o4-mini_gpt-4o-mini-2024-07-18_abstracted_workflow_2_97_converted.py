async def forward_97(self, taskInfo):

    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements

    # Sub-task 1: Analyze the given reaction components: starting material A, methyleneruthenium compound, and 1-propene
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given reaction components: starting material A, methyleneruthenium compound, and 1-propene. "
        "Classify their chemical nature and roles in the reaction context, focusing on the type of reaction likely involved."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                               model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing reaction components, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Analyze the product structure 1-(prop-1-en-1-yl)-2-vinylcyclopentane
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the analysis from Sub-task 1, analyze the product structure 1-(prop-1-en-1-yl)-2-vinylcyclopentane. "
        "Identify key structural features such as ring size, substituents, and connectivity to infer the type of transformation from starting material A."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                  model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}

    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing product structure, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    # Choose the most consistent answer from SC-CoT
    answer2_content_counts = Counter(possible_answers_2)
    most_common_answer2 = answer2_content_counts.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer2]
    answer2 = answermapping_2[most_common_answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Evaluate and Select Starting Material

    # Sub-task 3: Evaluate each choice by extracting structural features and comparing with inferred features
    debate_instruction_3 = (
        "Sub-task 3: Evaluate each choice (choice1 to choice4) by extracting their structural features and comparing them with the inferred features of starting material A "
        "based on the product analysis and reaction type."
    )
    debate_roles = ["Evaluator 1", "Evaluator 2"]
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                    model=self.node_model, role=role, temperature=0.5) 
                      for role in debate_roles]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]

    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating choices, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)

    # Final decision agent to select starting material
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                          model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking3[-1] + all_answer3[-1], 
                                                     "Sub-task 4: Select the starting material A from the given choices that best fits the structural and mechanistic requirements deduced from the reaction and product analysis.", 
                                                     is_sub_task=True)
    agents.append(f"Final Decision agent, selecting starting material, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
