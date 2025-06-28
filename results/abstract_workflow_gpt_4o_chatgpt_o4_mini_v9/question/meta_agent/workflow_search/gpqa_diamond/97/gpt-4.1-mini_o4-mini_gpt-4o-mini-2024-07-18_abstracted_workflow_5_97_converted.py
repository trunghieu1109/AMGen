async def forward_97(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Extract and characterize key structural features of product and reactants
    # Sub-task 1: Extract and characterize product features using Chain-of-Thought
    cot_instruction_1 = (
        "Sub-task 1: Extract and characterize the key structural features of the product 1-(prop-1-en-1-yl)-2-vinylcyclopentane, "
        "including the nature and position of substituents on the cyclopentane ring."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting product features, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 0.1: ", sub_tasks[-1])

    # Sub-task 2: Extract and characterize methyleneruthenium compound and 1-propene features using Chain-of-Thought
    cot_instruction_2 = (
        "Sub-task 2: Extract and characterize the structural features and reactivity of the methyleneruthenium compound and 1-propene as reactants, "
        "focusing on their roles in the reaction mechanism."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, extracting reactant features, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Analyze and classify candidate starting materials
    # Sub-task 3: Analyze and classify candidates using Self-Consistency Chain-of-Thought
    cot_sc_instruction_3 = (
        "Sub-task 3: Analyze and classify the four candidate starting materials by identifying their structural features, ring sizes, substituents, "
        "and unsaturation patterns relevant to the formation of the product, based on Sub-task 1 output."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, classifying candidates, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    most_common_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[most_common_answer_3]
    answer3 = answermapping_3[most_common_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 1.3: ", sub_tasks[-1])

    # Sub-task 4: Evaluate compatibility of candidates with reactants using Chain-of-Thought
    cot_instruction_4 = (
        "Sub-task 4: Evaluate the compatibility of each candidate starting material with the known reactivity of methyleneruthenium compounds and 1-propene, "
        "considering possible reaction pathways leading to the product, based on Sub-task 2 and Sub-task 3 outputs."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating candidate compatibility, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Propose reaction pathways and generate intermediates
    # Sub-task 5: Propose plausible reaction pathways using Debate
    debate_instruction_5 = (
        "Sub-task 5: Apply mechanistic reasoning to propose plausible reaction pathways from each candidate starting material with methyleneruthenium compound and 1-propene, "
        "predicting intermediate and final products, based on Sub-task 4 output."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, proposing pathways, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on plausible reaction pathways.", is_sub_task=True)
    agents.append(f"Final Decision agent on reaction pathways, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 2.5: ", sub_tasks[-1])

    # Sub-task 6: Generate structural variants/intermediates using Chain-of-Thought
    cot_instruction_6 = (
        "Sub-task 6: Generate structural variants or intermediates for each candidate starting material under the reaction conditions to assess which can yield the observed product structure, "
        "based on Sub-task 5 output."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, generating structural variants, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 2.6: ", sub_tasks[-1])

    # Stage 3: Evaluate and prioritize candidate starting materials
    # Sub-task 7: Evaluate and prioritize candidates using Reflexion
    cot_reflect_instruction_7 = (
        "Sub-task 7: Evaluate and prioritize the candidate starting materials based on their ability to produce the observed product 1-(prop-1-en-1-yl)-2-vinylcyclopentane "
        "through the proposed reaction pathways and mechanistic feasibility, using reflexion and critic feedback."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking6, answer6]
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, initial evaluation, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        feedback, correct = await critic_agent_7([taskInfo, thinking7, answer7], "Please review the candidate prioritization and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining evaluation round {i+1}, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 3.7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
