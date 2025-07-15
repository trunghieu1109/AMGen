async def forward_190(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    # Sub-task 1: SC_CoT to extract and summarize
    sc1_instruction = "Sub-task 1: Extract and summarize the functional groups of the starting material and outline each reaction step with its reagent and expected transformation."
    N1 = self.max_sc
    sc1_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    thinkings1 = []
    answers1 = []
    for agent in sc1_agents:
        thinking, answer = await agent([taskInfo], sc1_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, analyzing transformations, thinking: {thinking.content}; answer: {answer.content}")
        thinkings1.append(thinking)
        answers1.append(answer)
    final_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_agent1([taskInfo] + thinkings1 + answers1, "Sub-task 1: Synthesize and choose the most consistent summary of functional groups and steps.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append({"subtask_id":"subtask_1","instruction":sc1_instruction,"response":{"thinking":thinking1,"answer":answer1})
    print("Step 1: ", sub_tasks[-1])
    # Sub-task 2: SC_CoT to predict products 1 and 2
    sc2_instruction = "Sub-task 2: Using the summary from Sub-task 1, predict the structures of product 1 (O-benzyl ether formation) and product 2 (tosylhydrazone formation)."
    N2 = self.max_sc
    sc2_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    thinkings2 = []
    answers2 = []
    for agent in sc2_agents:
        thinking, answer = await agent([taskInfo, thinking1, answer1], sc2_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, predicting product1 & 2, thinking: {thinking.content}; answer: {answer.content}")
        thinkings2.append(thinking)
        answers2.append(answer)
    final_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_agent2([taskInfo, thinking1, answer1] + thinkings2 + answers2, "Sub-task 2: Synthesize and choose the most consistent structures for products 1 and 2.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append({"subtask_id":"subtask_2","instruction":sc2_instruction,"response":{"thinking":thinking2,"answer":answer2})
    print("Step 2: ", sub_tasks[-1])
    # Sub-task 3: SC_CoT to predict product 3 via Shapiro
    sc3_instruction = "Sub-task 3: Predict the structure of product 3 formed by the Shapiro reaction (n-BuLi induced elimination of the tosylhydrazone), using product 2 as input."
    N3 = self.max_sc
    sc3_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    thinkings3 = []
    answers3 = []
    for agent in sc3_agents:
        thinking, answer = await agent([taskInfo, thinking2, answer2], sc3_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, predicting product3, thinking: {thinking.content}; answer: {answer.content}")
        thinkings3.append(thinking)
        answers3.append(answer)
    final_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_agent3([taskInfo, thinking2, answer2] + thinkings3 + answers3, "Sub-task 3: Synthesize and choose the most consistent structure for product 3.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append({"subtask_id":"subtask_3","instruction":sc3_instruction,"response":{"thinking":thinking3,"answer":answer3})
    print("Step 3: ", sub_tasks[-1])
    # Sub-task 4: Debate for product4 and choice
    debate_instruction = "Sub-task 4: Determine the structure of product 4 after Pd/C hydrogenation (including saturation of all C=C bonds and hydrogenolysis of the benzyl ether), then select the matching multiple‐choice name. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    R = self.max_round
    all_thinking4 = [[] for _ in range(R)]
    all_answer4 = [[] for _ in range(R)]
    for r in range(R):
        for agent in debate_agents:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking3, answer3], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking, answer = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking4[r].append(thinking)
            all_answer4[r].append(answer)
    final_debate = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr4 = "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4, answer4 = await final_debate([taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1], final_instr4, is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append({"subtask_id":"subtask_4","instruction":debate_instruction,"response":{"thinking":thinking4,"answer":answer4})
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs