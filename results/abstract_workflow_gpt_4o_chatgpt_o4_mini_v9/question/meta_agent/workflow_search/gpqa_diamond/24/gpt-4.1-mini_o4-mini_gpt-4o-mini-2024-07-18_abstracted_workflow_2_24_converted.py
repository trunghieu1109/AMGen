async def forward_24(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements
    # Sub-task 1: Analyze the given chemical reactions and identify the type of name reactions involved
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given chemical reactions involving A + H2SO4 and B + BuLi + H+, "
        "identify the type of name reactions based on reactants, reagents, and products provided in the query. "
        "Focus on transformations implied by H2SO4, BuLi, and H+ and the product structures."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing name reactions, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Extract and classify structural features of the products
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on Sub-task 1 output, extract and classify the structural features of the products: "
        "2,8-dimethylspiro[4.5]decan-6-one and 4-methyl-1-phenylpent-3-en-1-ol. "
        "Identify key functional groups and ring systems to relate them to possible reactants."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, extracting product features, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer by frequency
    answer2_final = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_final]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Evaluate and Select Elements
    # Sub-task 3: Evaluate each choice of reactants A and B by comparing their structures and functional groups
    cot_instruction_3 = (
        "Sub-task 3: Evaluate each proposed choice of reactants A and B by comparing their structures and functional groups "
        "to the expected reactants that would yield the given products under the specified reaction conditions. "
        "Check for consistency with known name reactions and typical reactants."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2_final, answermapping_2[answer2_final]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, evaluating reactant choices, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Select the correct set of reactants (A and B) from the provided choices
    cot_reflect_instruction_4 = (
        "Sub-task 4: Based on evaluation in Sub-task 3, select the correct set of reactants (A and B) from the provided choices "
        "that best match the expected reactants for the given name reactions, considering structural and mechanistic consistency."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, selecting correct reactants, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback4, correct4 = await critic_agent_4([taskInfo, thinking4, answer4],
                                                 "Review the selection of correct reactants and provide limitations.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, feedback on reactant selection, thinking: {feedback4.content}; answer: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining reactant selection, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Final answer synthesis
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
