async def forward(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Stage 1: Condition Identification and Modular Reasoning

    # Sub-task 1: Determine speed s
    cot_instruction = "Sub-task 1: Determine the speed s in kilometers per hour based on the information that a 9 km walk at s km/h takes 4 hours minus t minutes."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {{cot_agent.id}}, determining speed s, thinking: {{thinking1.content}}; answer: {{answer1.content}}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {{thinking1.content}}; answer - {{answer1.content}}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Compute time t
    cot_sc_instruction = "Sub-task 2: Compute t, the time spent in the coffee shop, using the information from subtask 1 and the fact that at s + 2 km/h, the total time for 9 km walk is 2 hours and 24 minutes minus t minutes."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}

    for i in range(N):
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {{cot_agents[i].id}}, computing time t, thinking: {{thinking2.content}}; answer: {{answer2.content}}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2

    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {{thinking2.content}}; answer - {{answer2.content}}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Intermediate Inference and Validation

    # Sub-task 3: Calculate total time for Aya walking at speed s + 0.5 km/h
    cot_reflect_instruction = "Sub-task 3: Calculate the total time it will take for Aya walking at speed s + 0.5 km/h, incorporating previously determined values of t, s, and total distance."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round

    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {{cot_agent.id}}, calculating total time, thinking: {{thinking3.content}}; answer: {{answer3.content}}")

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking3, answer3], "please review the total time calculation and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {{critic_agent.id}}, providing feedback, thinking: {{feedback.content}}; answer: {{correct.content}}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {{cot_agent.id}}, refining total time calculation, thinking: {{thinking3.content}}; answer: {{answer3.content}}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {{thinking3.content}}; answer - {{answer3.content}}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 3: Final Output Generation and Integration

    # Sub-task 4: Convert time into minutes including time spent in coffee shop
    debate_instruction_4 = "Sub-task 4: Based on the output of sub-task 3, convert the total time including time in coffee shop into minutes and calculate the final answer."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round

    all_thinking4 = [[] for _ in range(N_max_3)]
    all_answer4 = [[] for _ in range(N_max_3)]

    for r in range(N_max_3):
        for agent in debate_agents_4:
            input_infos_4 = [taskInfo, thinking3, answer3]
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1])
            thinking4, answer4 = agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {{agent.id}}, round {{r}}, converting total time into minutes, thinking: {{thinking4.content}}; answer: {{answer4.content}}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)

    final_answers = [ans.content for ans in all_answer4[-1]]
    final_answercontent = Counter(final_answers).most_common(1)[0][0]
    index = final_answers.index(final_answercontent)
    thinking4 = all_thinking4[-1][index]
    answer4 = all_answer4[-1][index]
    sub_tasks.append(f"Sub-task 4 output: thinking - {{thinking4.content}}; answer - {{answer4.content}}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer