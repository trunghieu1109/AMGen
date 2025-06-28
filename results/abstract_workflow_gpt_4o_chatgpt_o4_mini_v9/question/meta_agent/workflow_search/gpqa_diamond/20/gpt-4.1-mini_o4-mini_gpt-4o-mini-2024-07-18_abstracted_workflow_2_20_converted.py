async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements

    # Sub-task 1: Analyze the concept of tautomerism
    cot_instruction_1 = (
        "Sub-task 1: Analyze the concept of tautomerism, including its definition, characteristics, "
        "and structural features that enable tautomerism. This foundational understanding is needed "
        "to evaluate whether benzoquinone and cyclohexane-1,3,5-trione can show tautomerism."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing tautomerism concept, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze molecular structures of benzoquinone and cyclohexane-1,3,5-trione for tautomerism
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the understanding of tautomerism from Sub-task 1, analyze the molecular structures "
        "and properties of benzoquinone and cyclohexane-1,3,5-trione to identify which compound can or cannot exhibit tautomerism."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing tautomerism in compounds, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinkingmapping_2[most_common_answer_2].content}; answer - {most_common_answer_2}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Analyze the concept of optical isomerism
    cot_instruction_3 = (
        "Sub-task 3: Analyze the concept of optical isomerism, including its definition, structural requirements "
        "such as chirality centers, and how it differs from other types of isomerism. This will help evaluate "
        "methyl 2-hydroxypropanoate and dimethyl fumarate for optical isomerism."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing optical isomerism concept, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Analyze molecular structures of methyl 2-hydroxypropanoate and dimethyl fumarate for optical isomerism
    cot_sc_instruction_4 = (
        "Sub-task 4: Based on the understanding of optical isomerism from Sub-task 3, analyze the molecular structures "
        "of methyl 2-hydroxypropanoate and dimethyl fumarate to determine which compound exhibits optical isomerism."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, analyzing optical isomerism in compounds, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    most_common_answer_4 = Counter(possible_answers_4).most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinkingmapping_4[most_common_answer_4].content}; answer - {most_common_answer_4}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Integrate results and select correct choice

    debate_instruction_5 = (
        "Sub-task 5: Integrate the results from Sub-task 2 (tautomerism evaluation) and Sub-task 4 (optical isomerism evaluation) "
        "to identify the correct choice among the given options (choice1 to choice4) that matches the compound that does not show tautomerism (A) "
        "and the compound that shows optical isomerism (B)."
    )
    debate_roles = ["Debater 1", "Debater 2"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinkingmapping_2[most_common_answer_2], answermapping_2[most_common_answer_2], thinkingmapping_4[most_common_answer_4], answermapping_4[most_common_answer_4]], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinkingmapping_2[most_common_answer_2], answermapping_2[most_common_answer_2], thinkingmapping_4[most_common_answer_4], answermapping_4[most_common_answer_4]] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating tautomerism and optical isomerism results, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct choice that matches the compound that does not show tautomerism (A) and the compound that shows optical isomerism (B).", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
