async def forward_16(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction = "Sub-task 1: Compute OI given R=13 and r=6 using OI squared equals R times (R minus 2r)"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking, answer = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, computing OI, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_agent = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_agent([taskInfo] + possible_thinkings + possible_answers, "Sub-task 1: synthesize and choose the most consistent answer for computing OI", is_sub_task=True)
    agents.append(f"Final Decision Agent, selecting OI, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Compute IA using OA=R and OI from Sub-task 1 via IA = sqrt(OA^2 - OI^2)"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking, answer = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, computing IA, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_agent = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_agent([taskInfo, thinking1, answer1] + possible_thinkings + possible_answers, "Sub-task 2: synthesize and choose the most consistent answer for computing IA", is_sub_task=True)
    agents.append(f"Final Decision Agent, selecting IA, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 3: Compute sin(A/2)=r/IA, then derive sinA=2 sin(A/2) cos(A/2), tan(A/2), cot(A/2)"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_sc_instruction,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking, answer = await cot_agents[i]([taskInfo, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, computing angles, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_agent = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_agent([taskInfo, thinking2, answer2] + possible_thinkings + possible_answers, "Sub-task 3: synthesize and choose the most consistent answer for angles", is_sub_task=True)
    agents.append(f"Final Decision Agent, selecting sinA and related values, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 4: Compute a=2R*sinA and semiperimeter s = a + r*cot(A/2)"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_sc_instruction,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking, answer = await cot_agents[i]([taskInfo, thinking3, answer3], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, computing side a and s, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_agent = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_agent([taskInfo, thinking3, answer3] + possible_thinkings + possible_answers, "Sub-task 4: synthesize and choose the most consistent answer for side a and s", is_sub_task=True)
    agents.append(f"Final Decision Agent, selecting side a and s, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 5: Compute AB*AC = 2*r*s/sinA"
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":cot_sc_instruction,"context":["user query","thinking of subtask 3","answer of subtask 3","thinking of subtask 4","answer of subtask 4"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking, answer = await cot_agents[i]([taskInfo, thinking3, answer3, thinking4, answer4], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, computing AB*AC, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_agent = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_agent([taskInfo, thinking3, answer3, thinking4, answer4] + possible_thinkings + possible_answers, "Sub-task 5: synthesize and choose the most consistent answer for AB*AC", is_sub_task=True)
    agents.append(f"Final Decision Agent, selecting AB*AC, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs