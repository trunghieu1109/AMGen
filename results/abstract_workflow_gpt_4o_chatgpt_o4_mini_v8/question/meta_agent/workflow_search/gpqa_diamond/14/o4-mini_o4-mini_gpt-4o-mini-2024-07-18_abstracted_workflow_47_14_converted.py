async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Extract Defining Features
    cot1_instr = "Sub-task 1: Extract orbital parameters P1 and P2 and note planet masses, with P1 three times shorter than P2."
    cot_agent1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent1([taskInfo], cot1_instr, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, extracting P1, P2 and mass info, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    sc_instr2 = "Sub-task 2: Extract M_star1, M_star2 and R_star1, R_star2 noting M_star1 = 2*M_star2 and R_star1 = R_star2."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers2 = []
    thinking_mapping2 = {}
    ans_mapping2 = {}
    for i in range(N2):
        t2, a2 = await cot_agents2[i]([taskInfo], sc_instr2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, extracting star parameters, thinking: {t2.content}; answer: {a2.content}")
        possible_answers2.append(a2.content)
        thinking_mapping2[a2.content] = t2
        ans_mapping2[a2.content] = a2
    answer2_content = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinking_mapping2[answer2_content]
    answer2 = ans_mapping2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Compute Expressions and Ratios
    cot3_instr = "Sub-task 3: Write expressions for semi-major axes a1 and a2 using Kepler's third law a = (G M_star P^2 / 4π^2)^(1/3)."
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1, thinking2, answer2], cot3_instr, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, deriving a1 and a2 expressions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    cot4_instr = "Sub-task 4: Derive symbolically the ratio a2/a1 = (M_star2 * P2^2 / (M_star1 * P1^2))^(1/3)."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot4_instr, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, deriving a2/a1, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    debate5_instr = "Sub-task 5: Using P_transit ≈ R_star/a and R_star1=R_star2, derive P_t1/P_t2 and simplify to show P_t1/P_t2 = a2/a1."
    debate_agents5 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N5 = self.max_round
    all_thinking5 = [[] for _ in range(N5)]
    all_answer5 = [[] for _ in range(N5)]
    for r in range(N5):
        for agent in debate_agents5:
            inputs5 = [taskInfo, thinking4, answer4]
            if r > 0:
                inputs5 += all_thinking5[r-1] + all_answer5[r-1]
            t5, a5 = await agent(inputs5, debate5_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, deriving transit ratio, thinking: {t5.content}; answer: {a5.content}")
            all_thinking5[r].append(t5)
            all_answer5[r].append(a5)
    final_decision5 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on P_t1/P_t2 expression.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision5.id}, deciding transit ratio, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    refl6_instr = "Sub-task 6: Substitute P1=P2/3 and M_star1=2*M_star2 into P_t1/P_t2 expression to compute numeric factor ~1.65."
    cot_agent6 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic6 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs6 = [taskInfo, thinking5, answer5]
    thinking6, answer6 = await cot_agent6(inputs6, refl6_instr, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent6.id}, computing numeric factor, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(self.max_round):
        fb6, corr6 = await critic6([taskInfo, thinking6, answer6], "Evaluate numeric computation correctness and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic6.id}, feedback on numeric computation, thinking: {fb6.content}; answer: {corr6.content}")
        if corr6.content == "True":
            break
        inputs6 += [thinking6, answer6, fb6]
        thinking6, answer6 = await cot_agent6(inputs6, refl6_instr, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent6.id}, refining numeric factor, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    cot7_instr = "Sub-task 7: Interpret result: since P_t1/P_t2 ≈1.65>1, Planet_1 has higher transit probability and matches choice1."
    cot_agent7 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent7([taskInfo, thinking6, answer6], cot7_instr, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent7.id}, interpreting numeric result, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer