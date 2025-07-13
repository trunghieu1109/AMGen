async def forward_24(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1 - Subtask 1 (SC_CoT)
    cot_sc1_instr = "Sub-task 1: Identify variables x, y, z and extract the three log2 equations accurately from the user query."
    cot_sc1_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    thoughts1 = []
    answers1 = []
    for agent in cot_sc1_agents:
        t, a = await agent([taskInfo], cot_sc1_instr, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {t.content}; answer: {a.content}")
        thoughts1.append(t)
        answers1.append(a)
    final_decider1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    t1, a1 = await final_decider1([taskInfo] + thoughts1 + answers1, "Sub-task 1 decision: Synthesize the correct extraction of variables and equations.", is_sub_task=True)
    sub_tasks.append(f"Stage1-Sub1 output: {a1.content}")
    logs.append({"subtask_id":"stage1_sub1","thinking":t1,"answer":a1})
    print("Step 1:", a1.content)

    # Stage 1 - Subtask 2 (CoT)
    cot2_instr = "Sub-task 2: Convert each log2 equation into its exponential form precisely."
    cot2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    t2, a2 = await cot2([taskInfo, t1, a1], cot2_instr, is_sub_task=True)
    agents.append(f"CoT agent {cot2.id}, thinking: {t2.content}; answer: {a2.content}")
    sub_tasks.append(f"Stage1-Sub2 output: {a2.content}")
    logs.append({"subtask_id":"stage1_sub2","thinking":t2,"answer":a2})
    print("Step 2:", a2.content)

    # Stage 2 - Subtask 1 (SC_CoT)
    sc2_instr = "Sub-task 1: Introduce exponents a, b, c with x=2^a, y=2^b, z=2^c and solve the resulting linear system."
    sc2_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    thoughts3 = []
    answers3 = []
    for agent in sc2_agents:
        t3, a3 = await agent([taskInfo, t2, a2], sc2_instr, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {t3.content}; answer: {a3.content}")
        thoughts3.append(t3)
        answers3.append(a3)
    final_decider3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    t3f, a3f = await final_decider3([taskInfo, t2, a2] + thoughts3 + answers3, "Sub-task 2 decision: Choose the correct a, b, c.", is_sub_task=True)
    sub_tasks.append(f"Stage2-Sub1 output: {a3f.content}")
    logs.append({"subtask_id":"stage2_sub1","thinking":t3f,"answer":a3f})
    print("Step 3:", a3f.content)

    # Stage 3 - Subtask 1 (Debate)
    debate3_instr = "Sub-task 1: Compute L = 4a+3b+2c and then |L|, given the solved values of a, b, c."
    debate3_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = []
    all_answer4 = []
    for agent in debate3_agents:
        t4, a4 = await agent([taskInfo, t3f, a3f], debate3_instr + " Given solutions to the problem from other agents, consider their opinions as additional advice.", is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, thinking: {t4.content}; answer: {a4.content}")
        all_thinking4.append(t4)
        all_answer4.append(a4)
    final_decider4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    t4f, a4f = await final_decider4([taskInfo, t3f, a3f] + all_thinking4 + all_answer4, "Sub-task 1 decision: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Stage3-Sub1 output: {a4f.content}")
    logs.append({"subtask_id":"stage3_sub1","thinking":t4f,"answer":a4f})
    print("Step 4:", a4f.content)

    # Stage 3 - Subtask 2 (Reflexion)
    reflex3_instr = "Sub-task 2: Simplify |L| into a reduced fraction m/n and verify gcd(m,n)=1. Given previous attempts and feedback, carefully consider where you could go wrong."
    cot_reflect = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_reflect([taskInfo, t4f, a4f], reflex3_instr, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_reflect.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    for _ in range(self.max_round):
        fb, corr = await critic([taskInfo, thinking5, answer5], "Please review and criticize the simplification. If correct, output exactly 'True' in 'correct'.", is_sub_task=True)
        agents.append(f"Critic agent {critic.id}, feedback: {fb.content}; correct: {corr.content}")
        if corr.content.strip() == "True":
            break
        thinking5, answer5 = await cot_reflect([taskInfo, thinking5, answer5, fb], reflex3_instr, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_reflect.id}, refined thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Stage3-Sub2 output: {answer5.content}")
    logs.append({"subtask_id":"stage3_sub2","thinking":thinking5,"answer":answer5})
    print("Step 5:", answer5.content)

    # Stage 4 - Subtask 1 (Debate)
    debate4_instr = "Sub-task 1: Compute the final integer m+n from the reduced m/n. Given solutions to the problem from other agents, consider their opinions as additional advice."
    debate4_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking6 = []
    all_answer6 = []
    for agent in debate4_agents:
        t6, a6 = await agent([taskInfo, thinking5, answer5], debate4_instr, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, thinking: {t6.content}; answer: {a6.content}")
        all_thinking6.append(t6)
        all_answer6.append(a6)
    final_decider6 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    t6f, a6f = await final_decider6([taskInfo, thinking5, answer5] + all_thinking6 + all_answer6, "Sub-task 1 decision: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Stage4-Sub1 output: {a6f.content}")
    logs.append({"subtask_id":"stage4_sub1","thinking":t6f,"answer":a6f})
    print("Step 6:", a6f.content)

    # Stage 4 - Subtask 2 (Reflexion)
    reflex4_instr = "Sub-task 2: Perform a final format check to ensure output is only an integer. Given previous attempts and feedback, carefully consider where you could go wrong."
    cot_reflect2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic2 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_reflect2([taskInfo, t6f, a6f], reflex4_instr, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_reflect2.id}, thinking: {thinking7.content}; answer: {answer7.content}")
    for _ in range(self.max_round):
        fb2, corr2 = await critic2([taskInfo, thinking7, answer7], "Please review and ensure output is a single integer. If correct, output exactly 'True' in 'correct'.", is_sub_task=True)
        agents.append(f"Critic agent {critic2.id}, feedback: {fb2.content}; correct: {corr2.content}")
        if corr2.content.strip() == "True":
            break
        thinking7, answer7 = await cot_reflect2([taskInfo, thinking7, answer7, fb2], reflex4_instr, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_reflect2.id}, refined thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Stage4-Sub2 output: {answer7.content}")
    logs.append({"subtask_id":"stage4_sub2","thinking":thinking7,"answer":answer7})
    print("Step 7:", answer7.content)

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs