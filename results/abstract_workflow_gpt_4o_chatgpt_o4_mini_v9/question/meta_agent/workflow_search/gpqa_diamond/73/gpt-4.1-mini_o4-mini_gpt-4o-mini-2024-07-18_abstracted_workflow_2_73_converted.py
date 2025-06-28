async def forward_73(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []
    
    # Stage 1: Analyze and classify problem elements
    cot_instruction_1 = (
        "Sub-task 1: Analyze and classify the given problem elements: the atom decay scenario, the known decay probability (32% within 100 minutes), "
        "and the elapsed time without decay (50 minutes). Identify the type of probability distribution that models the decay process (likely exponential decay) "
        "and clarify the meaning of the given probability in this context."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing problem elements, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    # Stage 1: Extract and transform decay probability into decay rate parameter
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, extract and transform the given decay probability (32% in 100 minutes) "
        "into the decay rate parameter (decay constant) of the exponential distribution, which will be used to calculate conditional probabilities for different time intervals."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, extracting decay rate, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    
    # Choose the most frequent answer from self-consistency
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2_content = thinkingmapping_2[answer2_content].content
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_content}; answer - {answer2_content}")
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # Stage 2: Evaluate conditional probability using decay rate and memoryless property
    cot_instruction_3 = (
        "Sub-task 3: Evaluate the conditional probability that the atom will decay in the next 150 minutes given that it has survived the first 50 minutes, "
        "using the decay rate parameter obtained in Sub-task 2 and the memoryless property of the exponential distribution."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answermapping_2[answer2_content]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, evaluating conditional decay probability, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    # Stage 2: Compare calculated probability with multiple-choice answers and select closest
    debate_instruction_4 = (
        "Sub-task 4: Compare the calculated conditional decay probability with the provided multiple-choice answers (52%, 48%, 44%, 40%) "
        "and select the closest matching option as the final answer."
    )
    debate_roles = ["Proponent", "Opponent"]
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
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
            agents.append(f"Debate agent {agent.id}, round {r}, comparing and selecting closest choice, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the closest matching multiple-choice answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final answer, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
