async def forward_161(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements

    # Sub-task 1: Analyze the given metric ds^2 = (32 / (4 - x^2 - y^2)) (dx^2 + dy^2) to identify its geometric meaning and domain constraints, including the interpretation of the radius r=2
    cot_instruction_1 = (
        "Sub-task 1: Analyze the metric ds^2 = (32 / (4 - x^2 - y^2)) (dx^2 + dy^2), "
        "determine its geometric meaning, domain constraints, and interpret the radius r=2 in this context."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing metric, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Classify the metric as a model of a pseudosphere or related surface, and determine the appropriate coordinate system and range for (x, y) consistent with radius r=2
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on Sub-task 1 output, classify the metric as a pseudosphere or related surface, "
        "determine the appropriate coordinate system and the valid range for (x, y) consistent with radius r=2."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, classifying metric, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer by majority vote
    counter_2 = Counter(possible_answers_2)
    best_answer_2 = counter_2.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Derive area element, set up integral, evaluate, and compare with choices

    # Sub-task 3: Derive the area element (dA) from the given metric by expressing the metric tensor components and computing the square root of its determinant
    cot_instruction_3 = (
        "Sub-task 3: Derive the area element dA from the metric ds^2 = (32 / (4 - x^2 - y^2)) (dx^2 + dy^2), "
        "express the metric tensor components and compute sqrt of its determinant."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, deriving area element, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Set up the integral for the total area of the pseudosphere of radius r=2 using the derived area element and the domain determined in previous subtasks
    cot_instruction_4 = (
        "Sub-task 4: Using the derived area element and domain from previous subtasks, set up the integral expression for the total area of the pseudosphere of radius r=2."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, setting up area integral, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Evaluate the area integral explicitly, simplifying the expression and computing the definite integral over the domain to find the total area
    debate_instruction_5 = (
        "Sub-task 5: Evaluate the area integral explicitly, simplify the expression, and compute the definite integral over the domain to find the total area."
    )
    debate_roles = ["Evaluator", "Simplifier"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
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
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating area integral, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the evaluated total area.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating total area, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Compare the computed area result with the given multiple-choice options and determine the correct choice
    cot_instruction_6 = (
        "Sub-task 6: Compare the computed total area with the given multiple-choice options: '+∞', '4π(x^2 + y^2)', '0', '4π(x^2 - y^2)', "
        "and determine the correct choice."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, comparing area with choices, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
