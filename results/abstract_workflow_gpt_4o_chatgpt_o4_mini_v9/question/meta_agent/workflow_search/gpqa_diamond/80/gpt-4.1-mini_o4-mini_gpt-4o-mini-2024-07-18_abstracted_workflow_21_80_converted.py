async def forward_80(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []
    
    # Stage 1: Analyze and Classify Elements
    
    # Sub-task 1: Analyze the target molecule [1,1'-bi(cyclopentylidene)]-2-one
    cot_instruction_1 = (
        "Sub-task 1: Analyze the target molecule [1,1'-bi(cyclopentylidene)]-2-one to understand its structural features, "
        "functional groups, and key bonds that must be formed during synthesis. This includes identifying the core ring system, "
        "the ketone group, and the bi-cyclopentylidene linkage."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing target molecule, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    
    # Sub-task 2: Analyze the starting material 1,5-dichloropentane
    cot_instruction_2 = (
        "Sub-task 2: Analyze the starting material 1,5-dichloropentane to identify its structure, reactive sites, "
        "and possible transformations that can lead to ring formation or functional group introduction relevant to the target molecule."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing starting material, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    
    # Sub-task 3: Analyze each reagent and condition in the provided choices
    cot_sc_instruction_3 = (
        "Sub-task 3: Analyze each reagent and condition in the provided choices (choice1 to choice4) to classify their chemical roles "
        "and predict their likely transformations on 1,5-dichloropentane or intermediates formed."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, analyzing reagents and conditions, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Select the most consistent answer by frequency
    answer_counter_3 = Counter(possible_answers_3)
    most_common_answer_3 = answer_counter_3.most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinkingmapping_3[most_common_answer_3].content}; answer - {most_common_answer_3}")
    print("Step 3: ", sub_tasks[-1])
    
    # Stage 2: Generate and Evaluate Variants
    
    # Sub-task 4: Generate plausible intermediate structures and reaction sequences
    cot_reflect_instruction_4 = (
        "Sub-task 4: Generate plausible intermediate structures and reaction sequences starting from 1,5-dichloropentane using the reagents in each choice, "
        "based on the chemical roles identified. This involves stepwise prediction of transformations such as ring closure, halogenation, oxidation, or reduction."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinkingmapping_3[most_common_answer_3], answermapping_3[most_common_answer_3]]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, generating intermediates and sequences, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4],
                                               "Critically evaluate the generated reaction sequences and intermediates for plausibility and consistency with organic synthesis principles.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining sequences, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    
    # Sub-task 5: Evaluate the plausibility of each reaction sequence
    debate_instruction_5 = (
        "Sub-task 5: Based on the output of Sub-task 4, evaluate the plausibility of each reaction sequence by comparing predicted intermediates and final products "
        "with the target molecule’s structure and functional groups, checking for consistency with known organic synthesis principles."
    )
    debate_roles_5 = ["Proponent", "Opponent"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles_5]
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
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating plausibility, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the plausibility evaluation.", is_sub_task=True)
    agents.append(f"Final Decision agent, evaluating plausibility, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    
    # Sub-task 6: Prioritize and select the correct sequence of reagents
    cot_instruction_6 = (
        "Sub-task 6: Prioritize and select the correct sequence of reagents from the given choices that most logically and feasibly leads to the synthesis of "
        "[1,1'-bi(cyclopentylidene)]-2-one starting from 1,5-dichloropentane, based on the evaluation in Sub-task 5."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, selecting correct reagent sequence, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
