async def forward_9(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    cot1_instruction = "Sub-task 1: Extract and tabulate the key attributes (mass, radius, density, composition assumption) for each planet: a) M=1 M⊕, R=1 R⊕; b) M=2 M⊕, density=5.5 g/cm^3; c) M=5 M⊕, Earth-like composition; d) M=0.5 M⊕, Earth-like composition."
    cot1_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot1_agent([taskInfo], cot1_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot1_agent.id}, extracting attributes, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    sc2_instruction = "Sub-task 2: Based on the extracted attributes, indicate for each planet whether density is already provided (b) or needs to be computed (a, c, d)."
    N = self.max_sc
    sc_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_map2 = {}
    answer_map2 = {}
    for i in range(N):
        thinking2_i, answer2_i = await sc_agents[i]([taskInfo, thinking1, answer1], sc2_instruction, is_sub_task=True)
        agents.append(f"SC-CoT agent {sc_agents[i].id}, determining density flags, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinking_map2[answer2_i.content] = thinking2_i
        answer_map2[answer2_i.content] = answer2_i
    final2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_map2[final2]
    answer2 = answer_map2[final2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    cot3_instruction = "Sub-task 3: Compute densities for planets a, c, and d by assuming Earth-like composition: compute R = M^(1/3) R⊕ and density = (M / R^3) * 5.5 g/cm^3 for each."
    cot3_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot3_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot3_agent(cot3_inputs, cot3_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot3_agent.id}, computing densities, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback3, correct3 = await critic_agent([taskInfo, thinking3, answer3], "Critically evaluate the density computations for correctness and completeness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback on computations, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        cot3_inputs.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot3_agent(cot3_inputs, cot3_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot3_agent.id}, refining computations, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    debate4_instruction = "Sub-task 4: Compare the densities of all four planets and determine which has the highest density."
    debate_agents4 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max4)]
    all_answer4 = [[] for _ in range(N_max4)]
    for r in range(N_max4):
        for i, agent in enumerate(debate_agents4):
            if r == 0:
                input4 = [taskInfo, thinking3, answer3]
            else:
                input4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
            thinking4, answer4 = await agent(input4, debate4_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing densities, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final4_agent = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final4_agent([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the highest density planet.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting highest density, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer