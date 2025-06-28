async def forward_85(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Analyze starting materials, products, and reducing agents independently using Chain-of-Thought (CoT)

    # Sub-task 1: Analyze the chemical structures and stereochemistry of the starting materials
    cot_instruction_1 = (
        "Sub-task 1: Analyze the chemical structures and stereochemistry of the given starting materials "
        "(the two unknown compounds) described as (S)-3-ethyl-5-isobutoxy-5-oxopentanoic acid and "
        "(R)-3-ethyl-5-isobutoxy-5-oxopentanoic acid, to identify their key functional groups and stereochemical centers relevant to the reactions."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing starting materials, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze the products formed in the reactions focusing on stereochemistry and structural features
    cot_instruction_2 = (
        "Sub-task 2: Analyze the products formed in the reactions: (R)-4-ethyltetrahydro-2H-pyran-2-one and "
        "(S)-4-ethyltetrahydro-2H-pyran-2-one, focusing on their stereochemistry and structural features, to understand the transformation from starting materials to products."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing products, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Identify and characterize the reducing agents LiBH4 and BH3 and their typical reaction mechanisms and stereochemical outcomes
    cot_instruction_3 = (
        "Sub-task 3: Identify and characterize the reducing agents used (LiBH4 and BH3) and their typical reaction mechanisms "
        "and stereochemical outcomes when reducing the functional groups present in the starting materials."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing reducing agents, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 1: Predict transformations and assess stereochemical relationships using Self-Consistency Chain-of-Thought (SC-CoT) and Debate

    # Sub-task 4: Predict how each reducing agent transforms the starting materials into the products including stereochemical configuration
    cot_sc_instruction_4 = (
        "Sub-task 4: Apply knowledge of the reducing agents mechanisms (from subtask_3) to predict how each reducing agent (LiBH4 and BH3) "
        "would transform the starting materials (from subtask_1) into the products (from subtask_2), including the expected stereochemical configuration of the products."
    )
    N = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, predicting transformations, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    # Choose the most frequent answer
    answer4_final = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4_final = thinkingmapping_4[answer4_final]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_final.content}; answer - {answer4_final}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Assess stereochemical relationship between starting materials and products for each reaction using Debate
    debate_instruction_5 = (
        "Sub-task 5: Assess the stereochemical relationship between the starting materials and the products for each reaction, "
        "determining which enantiomeric starting material corresponds to which product configuration after reduction by the respective reducing agent."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4_final, answermapping_4[answer4_final]], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4_final, answermapping_4[answer4_final]] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, assessing stereochemical relationships, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on stereochemical relationships.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding stereochemical relationships, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Evaluate choices against predicted outcomes using Reflexion

    cot_reflect_instruction_6 = (
        "Sub-task 6: Evaluate each provided choice of starting materials (choices 1 to 4) against the predicted stereochemical outcomes "
        "and reaction pathways established in subtask_5, to identify which choice correctly matches the starting materials A and B to their respective products after treatment with LiBH4 and BH3."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, evaluating choices, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "Critically evaluate the choice evaluation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining choice evaluation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
